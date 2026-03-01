# Description: UNet model for image or video generation.
# Inferences: 1. https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
#             2. https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py
#             3. https://github.com/tqch/ddpm-torch/blob/master/ddpm_torch/models/unet.py
#             4. https://github.com/hojonathanho/diffusion

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from functools import partial
from torch import Tensor
from typing import Optional, Tuple

from utils.helpers import default, exists, divisible_by, cast_tuple

#-----------------Embedding Blocks-----------------#
class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        """
        Sinusoidal positional encoding for time steps.
        Args:
            embedding_dim: Dimension of the positional embedding (must be even).
        """
        super(SinusoidalPosEmbedding, self).__init__()
        assert divisible_by(embedding_dim, 2), "Embedding dimension must be even."
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        Forward method for computing sinusoidal embeddings.
        Args:
            t: Input tensor of shape [batch_size] or [batch_size, 1].
        Returns:
            Sinusoidal embeddings of shape [batch_size, embedding_dim].

        here, min_scale = 1.0, max_scale = 10000.
        """
        half_dim = self.embedding_dim // 2
        # Compute the frequency factors
        emb = -math.log(10000.0) / (half_dim - 1) # -logterm = log(min_scale/max_scale)/(embedding_dim/2-1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=x.device) * emb) # 1/div_term = exp(k * -logterm)
        # freq = omega = 1/(min_scale * divterm) = 1/divterm = emb
        # Compute sinusoidal embeddings
        sinusoidal_emb = torch.cat([torch.sin(x[:, None] * emb), torch.cos(x[:, None] * emb)], dim=-1)
        return sinusoidal_emb

#----------------- Attention Blocks-----------------#    
class LinearAttentionBlock(nn.Module):
    def __init__(self, dim, channel, *, num_head=4, dim_head=32, num_mem_kv=4):
        super(LinearAttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_head
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5 # zoom factor
        hidden_dim = dim_head * num_head
        self.norm = Norm_Selected(channel)
        self.mem_kv = nn.Parameter(torch.randn(2, num_head, dim_head, num_mem_kv))
        self.to_qkv = conv_nd(
            dims=dim,
            in_channels=channel, 
            out_channels=hidden_dim * 3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias = False
        )
        self.to_out = nn.Sequential(
            conv_nd(
                dims=dim,
                in_channels=hidden_dim, 
                out_channels=channel, 
                kernel_size=1,
                stride=1,
                padding=0
                ),
            Norm_Selected(channel)
        )
    def forward(self, x):
        if self.dim == 1:
            b, c, n = x.shape
        elif self.dim == 2:
            b, c, h, w = x.shape
        elif self.dim == 3:
            b, c, d, h, w = x.shape
        else:
            assert False, f"Invalid dimension: {self.dim}"

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1) # 3 * (bs, hidden_dim, h, w)

        if self.dim == 1:
            q, k, v = map(lambda t: rearrange(t, "b (h d) x -> b h d x", h=self.num_heads), qkv) # 3 * (bs, num_head, dim_head, n)
        elif self.dim == 2:
            q, k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h d (x y)", h=self.num_heads), qkv)# 3 * (bs, num_head, dim_head, (h*w))
        elif self.dim == 3:
            q, k, v = map(lambda t: rearrange(t, "b (h d) x y z -> b h d (x y z)", h=self.num_heads), qkv)# 3 * (bs, num_head, dim_head, (d*h*w))

        mk, mv = map(lambda t: repeat(t, "h d n -> b h d n", b = b), self.mem_kv) # 2 * (bs, num_head, dim_head, num_mem_kv)

        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))# 2 * (bs, num_heads, dim_head, (h*w + num_mem_kv))

        # instead of softmax together as full, we tore the softmax of q and k separately
        q = q.softmax(dim=-2) # (bs, num_head, dim_head, (h*w)), needs -2 dim to multiply with kv
        k = k.softmax(dim=-1) # (bs, num_heads, dim_head, (h*w + num_mem_kv)), needs -1 dim to multiply with v
        q = q * self.scale # (bs, num_head, dim_head, (h*w))

        # linear attention prevents the quadratic memory complexity O(n^2) of full attention, only O(n)
        '''
        full attention: O(n^2 * d)
        q: (bs, num_head, dim_head, n)
        k: (bs, num_head, dim_head, n)
        qk: (bs, num_head, n, n) ~ O(n^2)
        v = (bs, num_head, dim_head, n)
        qkv: (bs, num_head, dim_head, n) ~ O(n^2 * dim_head)

        linear attention: O(n * d^2)
        k: (bs, num_head, dim_head, n + num_mem_kv)
        v: (bs, num_head, dim_head, n + num_mem_kv)
        kv: (bs, num_head, dim_head, dim_head) ~ O(dim_head^2)
        q: (bs, num_head, dim_head, n)
        qkv: (bs, num_head, dim_head, n) ~ O(n * dim_head^2)
        '''
        context = torch.einsum("bhdn,bhen -> bhde", k, v) # (bs, num_head, dim_head, dim_head) Context matrix, d==e
        out = torch.einsum("bhde,bhdn -> bhdn", context, q) # (bs, num_head, dim_head, (h*w)), d==e
        if self.dim == 1:
            out = rearrange(out, "b h d n -> b (h d) n", h=self.num_heads)
        elif self.dim == 2:
            out = rearrange(out, "b h d (x y) -> b (h d) x y", h=self.num_heads, x = h, y = w) # (bs, hidden_dim, h, w)
        elif self.dim == 3:
            out = rearrange(out, "b h d (x y z) -> b (h d) x y z", h=self.num_heads, x = d, y = h, z = w)
        return self.to_out(out)
    
class Attend(nn.Module):
    def __init__(self, *, dropout=0.1, flash=False, scale=None):
        super(Attend, self).__init__()
        self.dropout = dropout
        self.flash = flash
        self.scale = scale
        # nn.Dropout is a regularization technique for reducing overfitting in neural networks.
        # During training, randomly zeroes some of the elements of the input tensor with probability p.
        self.attn_dropout = nn.Dropout(dropout)
    def flash_attention(self, q, k, v):
        pass
    def forward(self, q, k, v):
        '''
        bs: batch size
        hds: heads 
        d: feature dimension (dim_head)
        n, i, j: sequence length (base sequence length, source, target)

        q: query, shape: (bs, hds, i, d)
        k: key, shape: (bs, hds, j, d)
        v: value, shape: (bs, hds, j, d)
        '''
        q_len, k_len, device = q.shape[-1], k.shape[-1], q.device
        if self.flash:
            return self.flash_attention(q, k, v)
        else:
            scale = default(self.scale, q.shape[-2] ** -0.5)
            similarity = torch.einsum(f"bhdi, bhdj -> bhij", q, k) * scale
            attn = similarity.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.einsum("bhij, bhdj -> bhdi", attn, v)
            return out
        
class FullAttentionBlock(nn.Module):
    def __init__(self, dim, channel, *, num_head=4, dim_head=32, num_mem_kv=4, flash=False):
        super(FullAttentionBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_head
        self.scale = dim_head ** -0.5
        hidden_dim = dim_head * num_head
        self.num_mem_kv = num_mem_kv
        self.flash = flash
    
        self.norm = Norm_Selected(channel)

        if not flash:
            self.attn = Attend(dropout=0.1, scale=self.scale)
            # self.attn = nn.MultiheadAttention(embed_dim=channel, num_heads=num_head, batch_first=True)
        else:
            # Warning! CUDA only!!!!!!!!!!!!!!!!
            from flash_attn.flash_attention import FlashAttention
            self.attn = FlashAttention(dim=channel, heads=num_head, num_mem_kv=num_mem_kv)

        self.mem_kv = nn.Parameter(torch.randn(2, num_head, dim_head, num_mem_kv))
        self.to_qkv = conv_nd(
            dims=dim,
            in_channels=channel, 
            out_channels=hidden_dim * 3, 
            kernel_size=1,
            stride=1,
            padding=0,
            bias = False)
        self.to_out = nn.Sequential(
            conv_nd(
                dims=dim,
                in_channels=hidden_dim, 
                out_channels=channel, 
                kernel_size=1,
                stride=1,
                padding=0),
            Norm_Selected(channel)
        )

    def forward(self, x):
        if self.dim == 1:
            b, c, n = x.shape
        elif self.dim == 2:
            b, c, h, w = x.shape
        elif self.dim == 3:
            b, c, d, h, w = x.shape
        else:
            assert False, f"Invalid dimension: {self.dim}"

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)

        if self.dim == 1:
            q, k, v = map(lambda t: rearrange(t, "b (h d) x -> b h d x", h=self.num_heads), qkv) # 3 * (bs, num_head, dim_head, n)
        elif self.dim == 2:
            q, k, v = map(lambda t: rearrange(t, "b (h d) x y -> b h d (x y)", h=self.num_heads), qkv)# 3 * (bs, num_head, dim_head, (h*w))
        elif self.dim == 3:
            q, k, v = map(lambda t: rearrange(t, "b (h d) x y z -> b h d (x y z)", h=self.num_heads), qkv)# 3 * (bs, num_head, dim_head, (d*h*w))
        '''
        bs: batch size
        hds: heads
        d: dim_head
        h: height
        w: width
        hidden_dim = heads * dim_head
        '''
        mk, mv = map(lambda t: repeat(t, "hds d nkv -> b hds d nkv", b = b), self.mem_kv)
        '''
        bs: batch size
        hds: heads
        nkv: num_mem_kv
        d: dim_head
        '''
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v))) # concatenate on memory key-value pairs
        out= self.attn(q, k, v)

        if self.dim == 1:
            out = rearrange(out, "b h d n -> b (h d) n", h=self.num_heads)
        elif self.dim == 2:
            out = rearrange(out, "b h d (x y) -> b (h d) x y", h=self.num_heads, x = h, y = w) # (bs, hidden_dim, h, w)
        elif self.dim == 3:
            out = rearrange(out, "b h d (x y z) -> b (h d) x y z", h=self.num_heads, x = d, y = h, z = w)
        return self.to_out(out)
    
class ToyAttentionBlock(nn.Module):
    def __init__(self, args, channel, num_head=4):
        super(ToyAttentionBlock, self).__init__()
        self.args = args
        self.norm = nn.LayerNorm([self.args.img_dim*self.args.img_dim, channel])
        self.attn = nn.MultiheadAttention(embed_dim=channel, num_heads=num_head, batch_first=True)

    def forward(self, x):
        # Flatten spatial dimensions for attention
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # Shape: [B, H*W, C]
        x_flat = self.norm(x_flat)
        attn_output, _ = self.attn(x_flat, x_flat, x_flat)  # Self-attention
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)  # Reshape back
        return x + attn_output

#-----------------Normalization Blocks-----------------#
class RMSNorm(nn.Module):

    '''
    It is more robust to offset. 
    The calculation cost is low, suitable for efficient models.
    However, it lacks of control over the mean, in some case it is not as good as layernorm
    '''

    def __init__(self, out_channel):
        super(RMSNorm, self).__init__()
        self.zoom_fac = out_channel ** 0.5
        self.gamma = nn.Parameter(torch.ones(1, out_channel, 1, 1))
    def forward(self, x):
        norm = F.normalize(input=x, p=2, dim=1, eps=1e-12) # x/RMS(x)
        return norm * self.gamma * self.zoom_fac
        # rms = x.pow(2).mean(dim=[1, 2, 3], keepdim=True).add(1e-12).sqrt()
        # x = x / rms
        # return x * self.gamma

class LayerNorm2D(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        # Rearrange to [B, H, W, C] for LayerNorm
        x = x.permute(0, 2, 3, 1)
        # Apply LayerNorm
        x = self.layer_norm(x)
        # Rearrange back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x

class Norm_Selected(nn.Module):
    def __init__(self, out_channel, norm_type='gn1', num_groups=32):
        super(Norm_Selected, self).__init__()
        self.norm_type = norm_type
        if norm_type == 'rms':
            self.norm = RMSNorm(out_channel)
        elif norm_type == 'ln':
            self.norm = LayerNorm2D(out_channel)
        elif norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_channel)
        elif norm_type == 'gn1':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channel)
        elif norm_type == 'gn2':
            self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channel)
        else:
            raise ValueError(f'Invalid norm type: {norm_type}')
    def forward(self, x):
        return self.norm(x)
    
#-----------------Utility Blocks-----------------#
def zero_module(module):
    """
    Zero out the parameters of a module and return it.

    For diffusion models, zero_module ensures that the output of a particular convolutional layer is zero, so that it does not interfere with network behavior in the initial phase.
    Used as the last layer of the model to ensure that the initial output is close to zero, conducive to gradient stabilization.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

#-----------------Unet Block Modules-----------------#    
class Block(nn.Module):
    def __init__(self, in_channel, out_channel, *,  activation=nn.SiLU(), dims=2, init_zero=False, dropout=0.1):
        super(Block, self).__init__()

        self.norm = Norm_Selected(in_channel) # The feature map is normalized by channel
        self.activation = activation # silu(x)=x*σ(x),where σ(x) is the logistic sigmoid.

        if init_zero:
            self.conv1 = zero_module(
                conv_nd(
                    dims,
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
        else:
            self.conv1 = conv_nd(
                            dims,
                            in_channels=in_channel,
                            out_channels=out_channel,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        ) # The in_channel is projected using a 3 x 3 convolution to generate a feature map with out_channel as the number of output channels.
        self.dropout = nn.Dropout(dropout) # Randomly zeroed partial activation values to prevent overfitting
    
    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.dropout(x)
        return x
    
class CondResBlock(nn.Module):
    '''
    for introducing time embedding with non-linearity and increasing the receptive field of the network
    '''

    def __init__(self, in_channel, out_channel, *, t_emb_dim=None, c_emb_dim=None, dims=2, dropout=0.1, scale_shift=True, conv_skip=False):
        super(CondResBlock, self).__init__()

        self.scale_shift = scale_shift

        self.block_in = Block(in_channel, out_channel, dims=dims, init_zero=False, dropout=dropout)
        
        emb_dim = (t_emb_dim or 0) + (c_emb_dim or 0)
        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channel*2 if scale_shift else out_channel) # 2*out_channel as scale and shift
        ) if exists(t_emb_dim or c_emb_dim) else None


        self.block_out = nn.Sequential(
            Norm_Selected(out_channel),
            nn.SiLU(),
            nn.Dropout(dropout),
            conv_nd(
                dims=dims,
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        if out_channel == in_channel:
            self.skip_connection = nn.Identity()
        elif conv_skip:
            self.skip_connection = conv_nd(
                dims=dims,
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1
            )
        else:
            self.skip_connection = conv_nd(
                dims=dims,
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x, t_emb=None, c_emb=None):
        '''
        shape is not changed.
        x: bs, c, h, w
        t_emb: bs, t_emb_dim
        '''
        h = self.block_in(x) # shape: (bs, c_in, h, w) -> (bs, c_out, h, w)

        emb = []
        if exists(t_emb):
            emb.append(t_emb)
        if exists(c_emb):
            emb.append(c_emb)
        emb = torch.cat(emb, dim=-1) if len(emb) > 0 else None

        if exists(self.emb_mlp) and exists(emb):
            emb = self.emb_mlp(emb)
            while emb.dim() < h.dim():
                emb = emb[..., None]
            if self.scale_shift:
                out_norm, out_rest = self.block_out[0], self.block_out[1:]
                scale, shift = emb.chunk(2, dim=1) # split the 2*out_channel as scale and shift, shape: (bs, ct/2, 1, 1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb
                h = self.block_out(h)

        x = self.skip_connection(x)
        return h + x # sum with residual
    
class DownBlock(nn.Module):
    def __init__(self, args):
        super(DownBlock, self).__init__()
        self.args = args
        self.down_layers = nn.ModuleList()

        num_stages = len(self.args.base_mults)
        full_attns = (*((False,) * (len(self.args.base_mults) - 1)), True) # layer attention is applied at the last stage, except for others are linear.
        full_attns = cast_tuple(full_attns, num_stages)
        num_heads = cast_tuple(self.args.num_head, num_stages)
        dim_heads = cast_tuple(self.args.dim_head, num_stages)

        channel_dims = [self.args.base_channel, *map(lambda m: self.args.base_channel * m, self.args.base_mults)]
        self.channel_dims = list(zip(channel_dims[:-1], channel_dims[1:])) # [(32, 32), (32, 64), (64, 128)]
        

        self.kernel_size_1 = 3 if args.dim_conv == 2 else (3, 3, 3)
        self.stride_1 = 2 if args.dim_conv == 2 else (1, 2, 2)
        self.padding_1 = 1 if args.dim_conv == 2 else (1, 1, 1)
        self.downsample1 = lambda dim_in, dim_out: nn.Sequential(
            conv_nd(
                dims=args.dim_conv,
                in_channels=dim_in, 
                out_channels=default(dim_out, dim_in), 
                kernel_size=self.kernel_size_1, 
                stride=self.stride_1,
                padding=self.padding_1
            ) 
        )

        self.kernel_size_2 = 3 if args.dim_conv == 2 else (3, 3, 3)
        self.stride_2 = 1 if args.dim_conv == 2 else (1, 1, 1)
        self.padding_2 = 1 if args.dim_conv == 2 else (1, 1, 1)
        self.downsample2 = lambda dim_in, dim_out: conv_nd(
            dims=args.dim_conv,
            in_channels=dim_in, 
            out_channels=dim_out, 
            kernel_size=self.kernel_size_2, 
            stride=self.stride_2, 
            padding=self.padding_2
        )

        for i, ((in_channel, out_channel), full_attn, num_head, dim_head) in enumerate(zip(self.channel_dims, full_attns, num_heads, dim_heads)):
            is_last = i == (num_stages - 1)
            # attn = LinearAttentionBlock
            attn = FullAttentionBlock if full_attn else LinearAttentionBlock
            self.down_layers.append(
                nn.ModuleList([
                    CondResBlock(in_channel, in_channel, t_emb_dim=self.args.t_emb_dim, dims=args.dim_conv), # skip connection by residual block
                    CondResBlock(in_channel, in_channel, t_emb_dim=self.args.t_emb_dim, dims=args.dim_conv), # skip connection by residual block
                    attn(args.dim_conv, in_channel, num_head=num_head, dim_head=dim_head),
                    self.downsample1(dim_in=in_channel, dim_out=out_channel) if not is_last else self.downsample2(dim_in=in_channel, dim_out=out_channel)
                ])
            )

    def forward(self, x, t, c):
        h = []
        for res1, res2, attn, downsample in self.down_layers:
            x = res1(x, t)
            h.append(x)
            x = res2(x, t) 
            x = attn(x) + x
            h.append(x)
            x = downsample(x)
        return x, h
        
class UpBlock(nn.Module):
    def __init__(self, args):
        super(UpBlock, self).__init__()
        self.args = args
        self.up_layers = nn.ModuleList()

        num_stages = len(self.args.base_mults)
        full_attns = (*((False,) * (len(self.args.base_mults) - 1)), True) # layer attention is applied at the last stage, except for others are linear.
        full_attns = cast_tuple(full_attns, num_stages)
        num_heads = cast_tuple(self.args.num_head, num_stages)
        dim_heads = cast_tuple(self.args.dim_head, num_stages)

        channel_dims = [self.args.base_channel, *map(lambda m: self.args.base_channel * m, self.args.base_mults)]
        channel_dims = list(zip(channel_dims[:-1], channel_dims[1:]))

        
        self.scale_factor = args.dim_conv if args.dim_conv == 2 else (1, 2, 2)
        self.kernel_size_1 = 3 if args.dim_conv == 2 else (3, 3, 3)
        self.stride_1 = 1 if args.dim_conv == 2 else (1, 1, 1)
        self.padding_1 = 1 if args.dim_conv == 2 else (1, 1, 1)
        self.upsample1 = lambda in_dim, out_dim: nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode='nearest'), # image is 2d so 2
            conv_nd(
                dims=args.dim_conv,
                in_channels=in_dim, 
                out_channels=default(out_dim, in_dim), 
                kernel_size=self.kernel_size_1, 
                stride=self.stride_1,
                padding=self.padding_1
                )
        )
        self.upsample2 = lambda in_dim, out_dim: conv_nd(
            dims=args.dim_conv,
            in_channels=in_dim, 
            out_channels=out_dim, 
            kernel_size=self.kernel_size_1, 
            stride=self.stride_1, 
            padding=self.padding_1
        )

        for i, ((in_channel, out_channel), full_attn, num_head, dim_head) in enumerate(zip(*map(reversed, (channel_dims, full_attns, num_heads, dim_heads)))):
            is_last = i == (num_stages - 1)
            attn = FullAttentionBlock if full_attn else  LinearAttentionBlock
            self.up_layers.append(
                nn.ModuleList([
                    CondResBlock(in_channel+out_channel, out_channel, t_emb_dim=self.args.t_emb_dim, dims=args.dim_conv),
                    CondResBlock(in_channel+out_channel, out_channel, t_emb_dim=self.args.t_emb_dim, dims=args.dim_conv),
                    attn(args.dim_conv, out_channel, num_head=num_head, dim_head=dim_head),
                    self.upsample1(in_dim=out_channel, out_dim=in_channel) if not is_last else self.upsample2(in_dim=out_channel, out_dim=in_channel)
                ])
            )

    def forward(self, x, h, t, c):
        for res1, res2, attn, upsample in self.up_layers:
            x = torch.concat([x, h.pop()], dim=1)
            x = res1(x, t)
            x = torch.concat([x, h.pop()], dim=1)
            x = res2(x, t)
            x = attn(x) + x
            x = upsample(x)
        return x
    
class BottleneckBlock(nn.Module):
    def __init__(self, args):
        super(BottleneckBlock, self).__init__()
        self.args = args
        self.bottleneck_dim = self.args.base_mults[-1] * self.args.base_channel
        self.res1 = CondResBlock(self.bottleneck_dim, self.bottleneck_dim, t_emb_dim=self.args.t_emb_dim, c_emb_dim=self.args.c_emb_dim, dims=args.dim_conv)
        self.att = FullAttentionBlock(args.dim_conv, self.bottleneck_dim, num_head=self.args.num_head, dim_head=self.args.dim_head)
        self.res2 = CondResBlock(self.bottleneck_dim, self.bottleneck_dim, t_emb_dim=self.args.t_emb_dim, c_emb_dim=self.args.c_emb_dim, dims=args.dim_conv)
    def forward(self, x, t, c):
        x = self.res1(x, t, c)
        x = self.att(x)
        x = self.res2(x, t, c)
        return x
    
class Bottleneck_UNet(nn.Module):
    '''
    Uniformly concatenate the same size of class embedding vector in each layer in Resnet.
    '''
    def __init__(self, args, self_cond=False, learn_var=False):
        super(Bottleneck_UNet, self).__init__()
        self.args = args

        self.t_embed = SinusoidalPosEmbedding(args.base_channel) 

        self.c_embed = nn.Linear(args.latent_dim, args.c_emb_dim)

        self.time_mlp = nn.Sequential(
            self.t_embed,
            nn.Linear(args.base_channel, args.t_emb_dim),
            nn.GELU(),
            nn.Linear(args.t_emb_dim, args.t_emb_dim)
        )
        self.cond_mlp = nn.Sequential(
            self.c_embed,
            nn.GELU(),
            nn.Linear(args.c_emb_dim, args.c_emb_dim)
        )


        init_channel = (self.args.image_channel*2) if self_cond else (self.args.image_channel)
        self.init_conv = conv_nd(
            dims=args.dim_conv, 
            in_channels=init_channel, 
            out_channels=self.args.base_channel, 
            kernel_size=3, 
            stride=1,
            padding=1) # combines the condition and img
        self.down = DownBlock(self.args)
        self.bottleneck = BottleneckBlock(self.args)
        self.up = UpBlock(self.args)
        final_channel = self.args.image_channel*2 if learn_var else self.args.image_channel
        self.final_res = CondResBlock(self.args.base_channel*2, self.args.base_channel, t_emb_dim=self.args.t_emb_dim, c_emb_dim=self.args.c_emb_dim, dims=args.dim_conv)
        self.final_conv = conv_nd(
            dims=args.dim_conv,
            in_channels=self.args.base_channel, 
            out_channels=final_channel, 
            kernel_size=1, 
            stride=1,
            padding=0)

    def forward(self, x, condition, t, x_self_cond=None):
        '''
        x: tensor of shape (bs, depth_channel, h, w, RGB), initial image
            MNIST: (bs, 1, 3, 32, 32) or (bs, 1, 32, 32)
        condition: tensor of shape (bs, latent_dim), latent condition
            MNIST: (bs, 1)
        t: int,random time step to sample, (bs,)
            MNIST: (bs,)
        '''

        orig_shape = x.shape

        assert all([d % (2**(len(self.down.down_layers))) == 0 for d in (x.shape[3:-1] if len(orig_shape) == 5 else x.shape[2:-1])]), "Image dimensions must be divisible by 2^num_downs"
        
        if self.args.self_cond:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat([x, x_self_cond], dim=1)

        if len(orig_shape) == 5:
            # Reshape input x to merge the 3 channels (3x3) into 9 channels            
            if self.args.dim_conv == 2:
                B, C, D, H, W = x.shape
                x = x.reshape(B, D*C, H, W)
            elif self.args.dim_conv == 3:
                B, C, D, H, W = x.shape
            else:
                raise ValueError(f"Invalid dim_conv: {self.args.dim_conv}")
            
        elif len(orig_shape) == 4:
            if self.args.dim_conv == 2:
                B, C, H, W = x.shape
            else: 
                raise ValueError(f"Invalid dim_conv: {self.args.dim_conv}")
        
        x = self.init_conv(x)  # shape: [B, 12, 160, 160] -> [B, 3, 160, 160]
        r = x.clone()
        t_emb = self.time_mlp(t) # shape: [B, t_emb_dim]
        c_emb = self.cond_mlp(condition)
        h = []
        x, h = self.down(x, t_emb, c_emb)
        x = self.bottleneck(x, t_emb, c_emb)
        x = self.up(x, h, t_emb, c_emb)
        x = torch.cat([x, r], dim=1)
        x = self.final_res(x, t_emb, c_emb)
        x = self.final_conv(x) # shape: [B, 3, 160, 160]
        if len(orig_shape) == 5:
            if self.args.dim_conv == 2:
                x = x.reshape(B, C, D, H, W)  # shape: [B, image_channels, 160, 160] -> [B, ts, rgb, 160, 160]
            elif self.args.dim_conv == 3:
                x = x
        elif len(orig_shape) == 4:
            if self.args.dim_conv == 2:
                x = x
            else:
                raise ValueError(f"Invalid dim_conv: {self.args.dim_conv}")
        return x