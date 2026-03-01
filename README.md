# PyTorch DDPM: Image & Multi-frame Generation

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-1.12+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-WIP-red.svg)

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) designed for flexible generation of both static images and dynamic multi-frame sequences (short videos). This project delves into various UNet architectures and advanced conditioning methods, with a core focus on adapting diffusion models for sequential data through flexible dimensionality.

## Features

-   **Core DDPM Implementation**: Complete forward (noising) and reverse (denoising) diffusion processes, adhering to the principles outlined in Ho et al., 2020.
-   **Multi-frame (Video-like) Generation**: Utilizes the `depth_channel` parameter to generate sequences of images, enabling the creation of short, animated outputs by treating multiple frames as a single input.
-   **Multiple UNet Architectures**:
    -   **Bottleneck UNet**: A standard UNet design with attention mechanisms strategically placed at the bottleneck and down/up sampling stages.
    -   **Early Fusion UNet**: Designed for optimized integration of conditional information early in the network layers.
    -   **Uniform UNet**: (Details to be confirmed, usually uniform attention/residual blocks).
    -   **OpenAI UNet**: Implements architecture concepts from OpenAI's improved diffusion models.
-   **Flexible Conditioning Mechanisms**: Supports integration of class embeddings, time embeddings, and self-conditioning for guided generation.
-   **Custom Noise Schedulers**: Implements flexible beta schedules (e.g., Linear, Cosine, Sigmoid variations) for controlling the diffusion process.
-   **Advanced Sampling**: Supports both standard DDPM reverse sampling and accelerated Denoising Diffusion Implicit Models (DDIM) sampling.
-   **Visualization Tools**: Utilities for plotting loss curves, visualizing forward/reverse diffusion steps, and calculating metrics.

## Algorithm Details

This section provides a deeper dive into the algorithmic components and architectural choices implemented in this project.

### 1. Denoising Diffusion Probabilistic Models (DDPM)

The core of this project is built upon the DDPM framework (Ho et al., 2020), which involves a fixed forward (noising) process and a learned reverse (denoising) process.

*   **Forward Diffusion Process**:
    Gradually adds Gaussian noise to an initial data point `x_0` over `T` timesteps, producing a sequence of noisy samples `x_1, ..., x_T`. This is a fixed Markov chain where `q(x_t|x_{t-1})` is defined by a variance schedule (`β_t`).
*   **Reverse Diffusion Process**:
    A neural network (UNet) is trained to denoise `x_t` back to `x_{t-1}` (or directly predict `x_0` or the noise `ε`), effectively reversing the noising process.
*   **Loss Function**:
    The model is typically trained using a Mean Squared Error (MSE) loss between the predicted noise and the true noise (`F.mse_loss`).
    *   **`min_SNR_lossweight`**: Supports a minimum Signal-to-Noise Ratio based loss weighting, which can improve training stability and sample quality, especially at very low noise levels.
    *   **Target Modes**: The UNet can be configured to predict `noise` (`ε`), the denoised image `x_0`, or an `intermediate variance`.
*   **Noise Schedules**: The `DiffusionModel` class supports flexible `beta` schedules (`self.alpha_cumprod_t`, `self.beta_t` are computed internally based on `num_ts` and potentially other parameters) which dictate the amount of noise added at each timestep.
*   **Sampling**:
    *   **DDPM Sampling**: Standard iterative reverse process, generating a sequence `x_T -> x_{T-1} -> ... -> x_0`.
    *   **DDIM Sampling**: Implemented via `ddim_sample` method, offering faster generation by allowing larger jumps between timesteps.

### 2. Conditional Generation

The model supports various methods to guide the generation process based on specific conditions:

*   **Time Embeddings (`t_emb`)**:
    *   Time steps `t` are transformed using **Sinusoidal Positional Embeddings** (`SinusoidalPosEmbedding`), a technique commonly used in Transformers to encode sequential information.
    *   These embeddings are then passed through a Multi-Layer Perceptron (`time_mlp`) before being integrated into the UNet layers.
*   **Class Embeddings (`c_emb`)**:
    *   Class labels (`cond`) are embedded into a dense vector space using a linear layer (`c_embed`) followed by an MLP (`cond_mlp`).
    *   These class embeddings are incorporated into the Conditional Residual Blocks (`CondResBlock`) within the UNet to guide generation towards specific classes (e.g., generating a specific digit or CIFAR-100 class).
*   **Self-Conditioning (`self_cond`)**:
    If enabled, the model feeds back a coarsely denoised version of `x_t` from the previous step as an additional input channel to the UNet for the current denoising step. This can improve sample quality and speed up convergence.

### 3. UNet Architectures & Transformer-inspired Attention

The project offers multiple UNet variants, with a strong focus on integrating **Transformer-inspired Attention Mechanisms** for enhanced feature learning, especially crucial for capturing long-range dependencies in complex images and multi-frame sequences.

*   **UNet Overview**:
    The UNet serves as the noise predictor. It comprises a symmetric encoder-decoder structure with skip connections, enabling it to capture both local fine-grained details and global contextual information.
*   **Attention Blocks**:
    Inspired by the success of Transformers in sequential data processing, this project incorporates specialized attention blocks within the UNet:
    *   **`LinearAttentionBlock`**: Implements a memory-efficient linear attention mechanism. Unlike standard self-attention which has quadratic complexity, linear attention reduces this to linear complexity, making it highly scalable for larger inputs like multi-frame sequences or higher resolution images. It is configured with parameters such as `num_head` (number of attention heads), `dim_head` (dimension of each head), and `num_mem_kv` (number of memory key-value pairs).
    *   **`FullAttentionBlock`**: Implements standard full self-attention. This block is typically employed in critical regions, such as the UNet's bottleneck, where capturing comprehensive global dependencies across the entire feature map is paramount. It can also be configured to use more advanced attention implementations like FlashAttention (if available and enabled) for further speedups on compatible hardware.
    *   **Integration**: These attention blocks are strategically inserted within the UNet's downsampling and upsampling paths, particularly at various resolution levels and within the bottleneck, to allow the model to dynamically focus on relevant parts of the input.
    *   **`einops.rearrange`**: The `einops` library is extensively used within these attention blocks for elegant and explicit tensor manipulations (e.g., `b (h d) x y -> b h d (x y)`), crucial for reshaping feature maps into suitable forms for multi-head attention computations.
*   **Conditional Residual Blocks (`CondResBlock`)**:
    These blocks are the fundamental building units of the UNet's encoder and decoder. They not only learn spatial features but also skillfully integrate time and class embeddings, supporting `scale_shift` normalization for powerful conditional modulation of learned features.
*   **Dimensionality (`dim_conv`)**:
    The `conv_nd` utility function and the overall UNet structure are designed for flexibility across different data modalities:
    *   **2D (`dim_conv=2`)**: Standard convolutional operations for typical image generation tasks.
    *   **3D (`dim_conv=3`)**: Extends convolutions to operate across an additional "depth" dimension, allowing the model to process and generate multi-frame sequences where depth corresponds to temporal frames.

## Directory Structure

```
DDPM_IMG/
├── src/                # Source code
│   ├── diffusion.py    # Main DDPM logic (DiffusionModel class)
│   ├── dataset.py      # Dataset loading and preprocessing
│   ├── config.py       # Configuration management (Config class)
│   ├── models/         # Neural network architectures (UNets and attention blocks)
│   └── utils/          # Helper functions and plotting tools
├── configs/            # YAML configuration files (e.g., cifar100.yaml, mnist.yaml)
├── notebooks/          # Jupyter notebooks for demos and inference
│   └── demo_inference.ipynb
├── results/            # Generated images, training logs, and model checkpoints
│   ├── CIFAR100_Results/
│   └── MNIST_Results/
├── scripts/            # Training and testing scripts
│   ├── train.py
│   └── run.sh          # Utility script for running train/sample/visualize modes
├── .gitignore          # Files ignored by Git (e.g., datasets, W&B logs, large models)
├── LICENSE             # Project license (MIT License)
└── requirements.txt    # Python dependencies
```

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/gengzhanguo/PyTorch-DDPM-Image-Video-Gen.git
    cd PyTorch-DDPM-Image-Video-Gen # Changed to the new repo name
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    (Optional: If using `conda`, activate your environment first: `source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate your_env_name`)

## Usage

This project uses `main.py` as a central entry point for training, sampling, and visualization, typically orchestrated by `run.sh` for convenience.

### General Options

All commands can be passed additional arguments to `main.py`. Common arguments include:

*   `--device [cuda|cpu|mps]`: Specify the compute device (default: `cuda` if available).
*   `--config_name [mnist|cifar100]`: Specify the base configuration to load from `configs/`.
*   `--unet_choice [bottleneck|earlyfusion|uniform|openai|test_unet]`: Select the UNet architecture.
*   `--depth_channel [int]`: Set the number of frames per data point for multi-frame generation (default: `1` for single image).
*   `--dataset_path [path]`: Specify the directory where datasets are stored or will be downloaded (default: `./dataset`).
*   `--save_dir [path]`: Specify the directory to save results and checkpoints (default: `./CIFAR100_Results` or `./MNIST_Results` based on dataset).

### 1. Training a Model

To train a DDPM model:

```bash
./run.sh train [additional_args_for_main.py]
```

**Example: Train Bottleneck UNet on CIFAR-100 for 200 epochs (Multi-frame, 3 frames)**

```bash
./run.sh train --num_epochs 200 --dataset CIFAR100 --unet_choice bottleneck --depth_channel 3
```

Models and training logs will be saved in the configured `save_dir` (e.g., `./CIFAR100_Results`).

### 2. Sampling Images/Multi-frames

To generate new samples from a trained model:

```bash
./run.sh sample <path_to_model_checkpoint.pth> [additional_args_for_main.py]
```

**Example: Sample from a trained CIFAR-100 multi-frame model**

```bash
./run.sh sample ./CIFAR100_Results/path/to/my_multiframe_model.pth --dataset CIFAR100 --unet_choice bottleneck --depth_channel 3
```

Generated samples will be saved in `[save_dir]/samples/`.

### 3. Visualizing Diffusion Processes

To visualize the forward and reverse (DDIM) diffusion processes:

```bash
./run.sh visualize <path_to_model_checkpoint.pth> [additional_args_for_main.py]
```

**Example: Visualize diffusion for a trained MNIST model**

```bash
./run.sh visualize ./MNIST_Results/path/to/my_model.pth --dataset MNIST --unet_choice bottleneck
```

### Jupyter Notebook Demo

Explore `notebooks/demo_inference.ipynb` for an interactive step-by-step demonstration of loading data, initializing models, and performing inference and visualization, including multi-frame outputs.

## Visual Results

Here are some sample generated images and loss curves from our experiments. For multi-frame outputs, **it is highly recommended to replace static images with animated GIFs** to better showcase the "video-like" generation capability.

### MNIST Digits Generation (Bottleneck2D UNet)

**Generated Samples:**
![MNIST Digit 0](MNIST_Results/Bottleneck2D/0.png)
![MNIST Digit 1](MNIST_Results/Bottleneck2D/1.png)
![MNIST Digit 2](MNIST_Results/Bottleneck2D/2.png)
_Sample MNIST digits generated by the DDPM._

**Training Loss Curve:**
![MNIST Loss Curve](<MNIST_Results/Bottleneck2D/loss curve.png>)
_Training loss over epochs for MNIST dataset._

### CIFAR-100 Animal Generation (Bottleneck_2D UNet)

**Generated Samples:**
![CIFAR100 Apple](CIFAR100_Results/Bottleneck_2D/apple_0.png)
![CIFAR100 Fish](CIFAR100_Results/Bottleneck_2D/aquarium_fish_0.png)
![CIFAR100 Baby](CIFAR100_Results/Bottleneck_2D/baby_0.png)
_Sample CIFAR-100 images generated by the DDPM._

**Training Loss Curve:**
![CIFAR100 Loss Curve](<CIFAR100_Results/Bottleneck_2D/loss curve.png>)
_Training loss over epochs for CIFAR-100 dataset._

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 gengzhanguo

## Acknowledgements and References

This project is built upon the foundational work and implements concepts from:

1.  [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
2.  [Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)
3.  **Code References:**
    *   [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) (MIT License) - Provided a strong foundation for DDPM implementation.
    *   [openai/improved-diffusion](https://github.com/openai/improved-diffusion) (MIT License) - Influenced UNet architecture designs and diffusion model best practices.
    *   [tqch/ddpm-torch](https://github.com/tqch/ddpm-torch) - Reference for various DDPM components.
4.  **Libraries:**
    *   [einops](https://github.com/arogozhnikov/einops) - For elegant tensor manipulations in attention blocks.
