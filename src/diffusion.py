from ast import arg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import random
from functools import partial
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt 
from collections import namedtuple

from einops import rearrange # Assuming einops is installed and not part of utils
from utils.helpers import default, identity, denormalize_data # Import specific helper functions 

#-----------------Noise Sceduler and Diffusion Model-----------------#
class DiffusionModel(nn.Module):
    def __init__(self, args, unet):
        super(DiffusionModel, self).__init__()
        self.args = args
        self.unet = unet
        self.num_ts = args.num_ts
        self.device = args.device

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        
        self.beta_t = self.sigmoid_beta_schedule(args.num_ts) # num_ts,

        self.alpha_t = 1.0 - self.beta_t # num_ts,
        self.alpha_cumprod_t = torch.cumprod(self.alpha_t, dim=0) # num_ts,
        self.alpha_cumprod_t_minus_1 = F.pad(self.alpha_cumprod_t[:-1], (1, 0), value=1.0) # pad the first element with 1.0: num_ts,
        # self.sigma_t = torch.sqrt((1 - self.alpha_cumprod[:-1]) / (1 - self.alpha_cumprod[1:]) * self.beta_t[1:])

        # q(x_t | x_{t-1}), for mathamatical stability, use exp(0.5 * log) = sqrt replace sqrt
        register_buffer("sqrt_alpha_cumprod_t", torch.sqrt(self.alpha_cumprod_t)) # num_ts,
        register_buffer("sqrt_alpha_cumprod_t_minus_1", torch.sqrt(self.alpha_cumprod_t_minus_1)) # num_ts,
        register_buffer("log_one_minus_alpha_cumprod_t", torch.log(1. - self.alpha_cumprod_t)) # num_ts,
        register_buffer("sqrt_one_minus_alpha_cumprod_t", torch.sqrt(1. - self.alpha_cumprod_t)) # num_ts,
        register_buffer("sqrt_recip_alpha_cumprod_t", torch.sqrt(1. / self.alpha_cumprod_t)) # num_ts,
        register_buffer("sqrt_recip_minus_one_alpha_cumprod_t", torch.sqrt((1. / self.alpha_cumprod_t - 1.))) # num_ts,   sqrt(1-a_bar_t)/sqrt(a_bar_t), 

        # posterior q(x_{t-1} | x_t, x_0)
        register_buffer("sigma_square_t", self.beta_t * (1. - self.alpha_cumprod_t_minus_1) / (1. - self.alpha_cumprod_t)) # num_ts,
        register_buffer("log_sigma_square_t", torch.log(self.sigma_square_t.clamp(min=1e-20))) # num_ts,
        register_buffer("sigma_t", torch.sqrt(self.sigma_square_t)) # num_ts, for numerical stability
        register_buffer("mu_t_coeff1", self.beta_t * self.sqrt_alpha_cumprod_t_minus_1 / (1. - self.alpha_cumprod_t)) # num_ts,
        register_buffer("mu_t_coeff2", (1. - self.alpha_cumprod_t_minus_1) * torch.sqrt(self.alpha_t) / (1. - self.alpha_cumprod_t)) # num_ts,

        self.immiscible = args.immiscible

        # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        self.offset_noise_strength = args.offset_noise_strength

        SNR = self.alpha_cumprod_t / (1. - self.alpha_cumprod_t) # Signal Noise Ratio

        # paper: Efficient Diffusion Training via Min-SNR Weighting Strategy
        # https://arxiv.org/abs/2303.09556
        maybe_clipped_SNR = self.SNR.clamp(max=5) if args.min_SNR_lossweight else SNR.clone()
        if args.target_mode == "noise":
            register_buffer("loss_weight", maybe_clipped_SNR / SNR) 
        elif args.target_mode == "x_0":
            register_buffer("loss_weight", maybe_clipped_SNR)
        elif args.target_mode == "inter_var":
            register_buffer("loss_weight", maybe_clipped_SNR / (1. + SNR))

        ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "x_0"])


        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        
    # ---------------------------Noise Scheduler---------------------------#
    def linear_beta_schedule(self, timesteps):
        scale = 1000 / timesteps
        betas = torch.linspace(scale * 0.0001, scale * 0.02, timesteps).to(self.args.device)
        return betas

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=self.args.device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def sigmoid_beta_schedule(self, timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
        """
        sigmoid schedule
        proposed in https://arxiv.org/abs/2212.11972 - Figure 8
        better for images > 64x64, when used during training
        """
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, dtype = torch.float32, device=self.args.device) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def extract(self, x, t, *, x_shape=None):
        if x_shape is None:
            x_shape = self.args.dataset_shape
        b, *_ = t.shape
        t = t.type(torch.int64)
        out = x.gather(dim=-1, index=t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def sample_noise(self, shape):
        x_1 = torch.randn(shape).to(self.device)
        # return torch.clip(x_1, -1, 1)
        return x_1

    # ---------------------------Forward Process---------------------------#
    def forward_diffusion(self, x_0, t):
        '''
        Reference: DDPM paper, equation 4
        params:
            x0: tensor of shape (bs, 3, 150, 150, 1), initial image
            t: int,random time step to sample, (bs,)
        '''
        noise = self.sample_noise(x_0.shape)
        # print("noise", noise.shape)
        if self.immiscible:
            # Optimal noise matching
            x_00, noiise = tuple(rearrange(teensor, "b ... -> b (...)") for teensor in (x_0, noise))
            dist = torch.cdist(x_00, noiise, p=2) # euclidean distance
            from scipy.optimize import linear_sum_assignment
            _, assign = linear_sum_assignment(dist.cpu())
            assign = torch.from_numpy(assign).to(self.device)
            noise = noise[assign]

        # Here is an example of how to extract the correct value from the buffer
        # as the replacement for the indexing operation in the original code
        c1 = self.extract(self.sqrt_alpha_cumprod_t, t, x_shape=x_0.shape)
        # c1 = torch.sqrt(self.alpha_cumprod_t[t]).view(-1, 1, 1, 1, 1)
        # print("sqrt_alpha_cumprod_t", c1.shape)
        c2 = self.extract(self.sqrt_one_minus_alpha_cumprod_t, t, x_shape=x_0.shape)
        # c2 = torch.sqrt(1 - self.alpha_cumprod_t[t]).view(-1, 1, 1, 1, 1)
        # print("sqrt_one_minus_alpha_cumprod_t", c2.shape)
        x_t = c1 * x_0 + c2 * noise
        return x_t, noise

    # ---------------------------Reverse Process---------------------------#
    def reverse_diffusion(self, x_t, condition, t, *, clip_x0=True, rederive_pred_noise=False):
        """ Perform reverse diffusion to get x_{t-1} from x_t. """
        t = torch.tensor([t] * x_t.size(0), device=self.device, dtype=torch.long) # [1]
        # print("t", t.shape)
        maybe_clip_x0 = partial(torch.clip, min=self.args.RGB_range[0], max=self.args.RGB_range[1]) if clip_x0 else identity
        # Predict noise
        unet_pred = self.unet(x_t, condition, t)
        # print("pred_noise", pred_noise.shape)
        if self.args.target_mode == "noise":
            pred_noise = unet_pred
            # x_0 = sqrt(1/a_bar_t) * x_t - sqrt(1-a_bar_t)/sqrt(a_bar_t) * pred_noise
            x_0 = self.extract(self.sqrt_recip_alpha_cumprod_t, t) * x_t - \
                self.extract(self.sqrt_recip_minus_one_alpha_cumprod_t, t) * pred_noise
            x_0 = maybe_clip_x0(x_0)
            if clip_x0 and rederive_pred_noise:
                pred_noise = (self.extract(self.sqrt_recip_alpha_cumprod_t, t) * x_t - x_0) / \
                                self.extract(self.sqrt_recip_minus_one_alpha_cumprod_t, t)
        elif self.args.target_mode == "x_0":
            x_0 = unet_pred
            x_0 = maybe_clip_x0(x_0)
            pred_noise = (self.extract(self.sqrt_recip_alpha_cumprod_t, t) * x_t - x_0) / \
                            self.extract(self.sqrt_recip_minus_one_alpha_cumprod_t, t)
        elif self.args.target_mode == "inter_var":
            v = unet_pred
            x_0 = self.extract(self.sqrt_alpha_cumprod_t, t) * x_t - \
                    self.extract(self.sqrt_one_minus_alpha_cumprod_t, t) * v
            x_0 = maybe_clip_x0(x_0)
            pred_noise = (self.extract(self.sqrt_recip_alpha_cumprod_t, t) * x_t - x_0) / \
                            self.extract(self.sqrt_recip_minus_one_alpha_cumprod_t, t)
    
        # Predict x0 for debugging or visualization
        # x0 = self.predict_x0(x_t, pred_noise, t)
    

        # print("x0", x0.shape)
        mean = self.extract(self.mu_t_coeff1, t) * x_0 + \
                self.extract(self.mu_t_coeff2, t) * x_t # way 1, same as the other way.
        # mean = (x_t - (1 - alpha_t) * pred_noise / torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)
        # mean = (x_t -  (self.extract(self.beta_t, t) * pred_noise / self.extract(self.sqrt_one_minus_alpha_cumprod_t, t))) / torch.sqrt(self.extract(self.alpha_t, t))
        sigma = self.extract(self.sigma_t, t)
        # Add noise for t > 0
        eps = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        pred_x_t_minus_1 = mean + sigma * eps
        # pred_x_t_minus_1 = mean + self.extract(self.beta_t, t) * eps

        return pred_noise, pred_x_t_minus_1, x_0
    
    # @torch.inference_mode()
    def sample(self, condition, shape):
        x_t = self.sample_noise(shape)
        x_t_list = []  # Store x0 predictions for analysis
        x_0_list = []
        # pred_noise_list = []
        sample_ts = np.linspace(self.args.num_ts - 1, 0, self.args.num_ts).astype(int)
        with torch.no_grad():
            x_0 = None
            # for t in tqdm(reversed(range(self.num_ts))):
            for t in tqdm((sample_ts)):
                x_self_cond = x_0 if self.args.self_cond else None
                pred_noise, x_t, x_0 = self.reverse_diffusion(x_t, condition, t)
                x_t_list.append(x_t.cpu())  # Optionally save x0 for visualization
                x_0_list.append(x_0.cpu())
                # pred_noise_list.append(pred_noise.cpu())
                if any(torch.isnan(x_t).flatten()):
                    print(f"NaN detected at time step {t}")
                    break

        x_t = torch.stack(x_t_list, dim=0)
        x_0 = torch.stack(x_0_list, dim=0)

        x_t = denormalize_data(x_t)
        x_0 = denormalize_data(x_0)
        return x_t.cpu(), x_0.cpu()
    
    def ddim_sample(self, condition, shape, num_ts=11, eta=0.2):
        B = shape[0]
        
        sample_ts = np.linspace(self.args.num_ts - 1, 0, num_ts).astype(int)
        # sample_ts = list(range(0, self.args.num_ts, self.args.num_ts // num_pts))
        # sample_ts.append(self.args.num_ts - 1)
        # sample_ts = np.array(list(reversed(sample_ts)))
        sample_ts_pairs = list(zip(sample_ts[:-1], sample_ts[1:]))
        
        x_t = self.sample_noise(shape)
        x_t_list = [x_t.cpu()]
        x_0_list = [x_t.cpu()]

        x_0 = None
        with torch.no_grad():
            for t_n, t_n_minus_1 in tqdm(sample_ts_pairs, desc = "DDIM Sampling"):
                x_self_cond = x_0 if self.args.self_cond else None
                # Note! we are not using x_t prediction here!
                pred_noise, _, x_0 = self.reverse_diffusion(x_t, condition, t_n, clip_x0=True, rederive_pred_noise=True)

                if t_n_minus_1 < 0:
                    x_t = x_0
                    x_t_list.append(x_t.cpu())
                    x_0_list.append(x_0.cpu())
                    continue

                alpha_t = self.alpha_cumprod_t[t_n]
                alpha_t_minus_1 = self.alpha_cumprod_t[t_n_minus_1]
                sigma_t = eta * ((1 - alpha_t / alpha_t_minus_1) * (1 - alpha_t_minus_1) / (1 - alpha_t)).sqrt()
                c = (1. - alpha_t_minus_1 - sigma_t ** 2).sqrt()

                noise = torch.randn_like(x_t)

                x_t = x_0 * alpha_t_minus_1.sqrt() + c * pred_noise + sigma_t * noise

                x_t_list.append(x_t.detach().cpu())
                x_0_list.append(x_0.detach().cpu())
                
        x_t = torch.stack(x_t_list, dim=0)
        x_t = denormalize_data(x_t)
        x_0 = torch.stack(x_0_list, dim=0)
        x_0 = denormalize_data(x_0)
        return x_t.cpu(), x_0.cpu()


    # ---------------------------Loss Function---------------------------#
    def loss_function(self, x_0, cond, *, offset_noise_strength=None):
        if self.args.depth_channel == 1:
            # Squeeze the depth dimension (index 2) before unpacking
            B, C, H, W = x_0.squeeze(2).shape
            x_0 = x_0.squeeze(2) # Apply squeeze to x_0 itself
        else:
            # If depth_channel > 1, keep 5 dimensions
            B, C, D, H, W = x_0.shape
        x_0 = x_0.to(self.device) # Ensure x_0 is on device after potential squeezing
        cond = cond.to(self.device)  # [B, 1]
        # Sample random timesteps
        t = torch.randint(0, self.num_ts, (x_0.size(0),), device=self.device)
        
        x_t, noise = self.forward_diffusion(x_0, t)

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            if self.args.depth_channel > 1:
                offset_noise = torch.randn((B,C,D), device = self.device)
                noise += offset_noise_strength * rearrange(offset_noise, 'B C D -> B C D 1 1')
            else:
                offset_noise = torch.randn((B,C), device = self.device)
                noise += offset_noise_strength * rearrange(offset_noise, 'B C -> B C 1 1')

        # x_self_cond = None # This is used when there is no condition and the model is self conditioned
        # if self.args.self_cond and random.random() < 0.5:
        #     with torch.no_grad():
        #         pred_noise, x_self_cond, x_0 = self.reverse_diffusion(x_t, cond, t.numpy()) # x_0
        #         x_self_cond.detach_()

        unet_pred = self.unet(x_t, cond, t)

        if self.args.target_mode == "noise":
            pred_target = noise
        elif self.args.target_mode == "x_0":
            pred_target = x_0
        elif self.args.target_mode == "inter_var":
            inter_var = self.extract(self.sqrt_alpha_cumprod_t, t) * x_0 - \
                            self.extract(self.sqrt_one_minus_alpha_cumprod_t, t) * noise   
            pred_target = inter_var
        else: 
            pred_target = None
            print("Invalid target mode")
        
        loss = F.mse_loss(unet_pred, pred_target, reduction='mean')
        loss = loss * self.extract(self.loss_weight, t, x_shape=loss.shape)
        return loss.mean()
    