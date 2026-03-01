from matplotlib import pyplot as plt
import numpy as np
import torch


#-----------------Plotting-----------------#
def compare_beta_schedules(args, model):
    """ Compare different beta schedules. """
    timesteps = args.num_ts

    # Generate beta schedules
    beta_linear = model.linear_beta_schedule(timesteps).detach().cpu().numpy()
    beta_cosine = model.cosine_beta_schedule(timesteps).detach().cpu().numpy()
    beta_sigmoid = model.sigmoid_beta_schedule(timesteps).detach().cpu().numpy()
    
    print("beta_linear", beta_linear.shape)
    print("beta_cosine", beta_cosine.shape)
    print("beta_sigmoid", beta_sigmoid.shape)
    # Compute alpha and alpha_cumprod for all schedules
    linear_alpha = 1 - beta_linear
    linear_alpha_cumprod = np.cumprod(linear_alpha)
    linear_alpha_cumprod_t_1 = np.pad(linear_alpha_cumprod[:-1], (1, 0), constant_values=1.0)
    linear_sigma = beta_linear * (1 - linear_alpha_cumprod_t_1) / (1 - linear_alpha_cumprod)
    linear_sqrt_sigma = np.sqrt(linear_sigma)

    cosine_alpha = 1 - beta_cosine
    cosine_alpha_cumprod = np.cumprod(cosine_alpha)
    cosine_alpha_cumprod_t_1 = np.pad(cosine_alpha_cumprod[:-1], (1, 0), constant_values=1.0)
    cosine_sigma = beta_cosine * (1 - cosine_alpha_cumprod_t_1) / (1 - cosine_alpha_cumprod)
    cosine_sqrt_sigma = np.sqrt(cosine_sigma)

    sigmoid_alpha = 1 - beta_sigmoid
    sigmoid_alpha_cumprod = np.cumprod(sigmoid_alpha)
    sigmoid_alpha_cumprod_t_1 = np.pad(sigmoid_alpha_cumprod[:-1], (1, 0), constant_values=1.0)
    sigmoid_sigma = beta_sigmoid * (1 - sigmoid_alpha_cumprod_t_1) / (1 - sigmoid_alpha_cumprod)
    sigmoid_sqrt_sigma = np.sqrt(sigmoid_sigma)

    # Create subplots: 1x3 grid
    plt.figure(figsize=(18, 6))

    # Plot beta schedules
    plt.subplot(1, 3, 1)
    plt.plot(beta_linear, label='Linear Beta')
    plt.plot(beta_cosine, label='Cosine Beta', linestyle='--')
    plt.plot(beta_sigmoid, label='Sigmoid Beta', linestyle='-.')
    plt.xlabel('Time Step')
    plt.ylabel('Beta')
    plt.title('Beta Schedule')
    plt.legend()

    # Plot cumulative alpha schedules
    plt.subplot(1, 3, 2)
    plt.plot(linear_alpha_cumprod, label='Linear Alpha Cumulative', color='blue')
    plt.plot(cosine_alpha_cumprod, label='Cosine Alpha Cumulative', color='orange', linestyle='--')
    plt.plot(sigmoid_alpha_cumprod, label='Sigmoid Alpha Cumulative', color='green', linestyle='-.')
    plt.xlabel('Time Step')
    plt.ylabel('Alpha Cumulative Product')
    plt.title('Cumulative Alpha Schedule')
    plt.legend()

    # Plot 1/sqrt(alpha_t)
    plt.subplot(1, 3, 3)
    plt.plot(linear_sqrt_sigma, label='sqrt_sigma(Linear Alpha)', color='blue')
    plt.plot(cosine_sqrt_sigma, label='sqrt_sigma(Cosine Alpha)', color='orange', linestyle='--')
    plt.plot(sigmoid_sqrt_sigma, label='sqrt_sigma(Sigmoid Alpha)', color='green', linestyle='-.')
    plt.xlabel('Time Step')
    plt.ylabel('Sigma_t')
    plt.title('Sigma_t')
    plt.legend()

    # Show plots
    plt.tight_layout()
    plt.show()

def compute_plot_wasserstein_kl(args, model, data):
    """
    Compute Wasserstein distances and KL divergences for given samples.
    
    Args:
        samples (torch.Tensor): Tensor of shape [num_samples, 1, 32, 32].
        
    Returns:
        tuple: A tuple containing two lists - Wasserstein distances and KL divergences.
    """

    indices = generate_plot_ts(args, args.num_ts)
    # Store results
    samples = []

    # Ensure x0 has shape [1, C, H, W]
    x0 = torch.as_tensor(data[0].unsqueeze(0)).to(args.device)  
    print("x0 shape:", x0.shape)
    print("x0 range:", x0.min().item(), x0.max().item())

    # Forward diffusion
    with torch.no_grad():
        for t in indices:
            t_tensor = torch.as_tensor([t], device=args.device)
            x_t, noise = model.forward_diffusion(x0, t_tensor)
            # Append the first sample (index 0) from the batch to the results
            samples.append(x_t.cpu())

    print("Number of results:", len(samples))

    samples = torch.stack(samples, dim=0)

    print("Results shape:", samples.shape)
    # Remove unnecessary dimensions
    samples = samples.squeeze(dim=(1, -1))  # [num_samples, 32, 32]

    wasserstein_distances = []
    kl_divs = []

    from torch.distributions.normal import Normal
    normal_dist = Normal(0, 1)  # Standard normal distribution

    for sample in samples:
        # Remove channel dimension, get shape [32, 32]
        wasserstein_sample = sample.squeeze(dim=0)

        # Compute Wasserstein distance
        wasserstein_distance = torch.mean(torch.abs(wasserstein_sample - normal_dist.sample(wasserstein_sample.shape)))

        # Compute KL divergence
        mu = sample.mean()
        sigma = sample.std()
        kl_div = torch.log(sigma) + 0.5 * (mu ** 2 + sigma ** 2 - 1)

        wasserstein_distances.append(wasserstein_distance.item())
        kl_divs.append(kl_div.item())
    def plot_wasserstein_kl(wasserstein_distances, kl_divs):
        """
        Plot Wasserstein distances and KL divergences.
        
        Args:
            wasserstein_distances (list): List of Wasserstein distances.
            kl_divs (list): List of KL divergences.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Wasserstein distances
        ax[0].plot(wasserstein_distances)
        ax[0].set_title("Wasserstein Distances")
        ax[0].set_xlabel("Sample Index")
        ax[0].set_ylabel("Wasserstein Distance")

        # Plot KL divergences
        ax[1].plot(kl_divs)
        ax[1].set_title("KL Divergences")
        ax[1].set_xlabel("Sample Index")
        ax[1].set_ylabel("KL Divergence")

        plt.tight_layout()
        plt.show()

    # print("Wasserstein distances for each sample:", wasserstein_distances)
    # print("Length of distances list:", len(wasserstein_distances))
    # print("KL divergences for each sample:", kl_divs)
    # print("Length of KL divergences list:", len(kl_divs))

    plot_wasserstein_kl(wasserstein_distances, kl_divs)

    return wasserstein_distances, kl_divs

def generate_plot_ts(args, num_pts):
    indices_to_visualize = np.linspace(0, args.num_ts-1, num_pts).astype(int)
    indices_to_visualize = indices_to_visualize
    return indices_to_visualize

from .helpers import denormalize_data

def plot_diffusion(args, img_series, n_cols=11, *, reverse=False):
    '''
    img_series: tensor of shape (time, num_pt, rgb_channel, depth_channel, img_dim, img_dim)
    here num_pt = 1 forced
    '''
    print("img_series shape:", img_series.shape)
    if img_series.ndim == 6:

        img_series = img_series.permute(0, 1, 3, 4, 5, 2) # [T, 1, D, H, W, C]
        img_series = img_series if not reverse else img_series.flip(0)
        n_rows = args.depth_channel

        indices_to_visualize = generate_plot_ts(args, n_cols)
        # indices_to_visualize = indices_to_visualize if not reverse else list(reversed(indices_to_visualize))
        print("indices_to_visualize", indices_to_visualize)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

        if n_rows == 1:
            for col, t_index in enumerate(indices_to_visualize):
                img = img_series[t_index, 0, 0].numpy()
                # img = (img - img.min()) / (img.max() - img.min())
                if args.RGB_channel == 1:
                    axs[col].imshow(img, cmap="gray") # 
                else: 
                    axs[col].imshow(img)
                axs[col].axis("off")
        else:
            for col, t_index in enumerate(indices_to_visualize):
                for row in range(n_rows):  # Iterate over channels (rows)
                    # Select image channel
                    img = img_series[t_index, 0, row].numpy()  # Shape: [H, W, 3]

                    # Normalize and clip for visualization
                    # img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]

                    # Plot
                    if args.RGB_channel == 1:
                        axs[row, col].imshow(img, cmap="gray" if n_rows == 1 else None)
                    else:
                        axs[row, col].imshow(img)
            
                    axs[row, col].axis("off")

                    # Add title for the top row
                    if row == 0:
                        axs[row, col].set_title(f"t={t_index}")

        plt.tight_layout()
        plt.show()

    elif img_series.ndim == 5:
        img_series = img_series.permute(0, 1, 3, 4, 2)  # [T, 1, H, W, C]
        img_series = img_series if not reverse else img_series.flip(0)
        n_rows = 1
        indices_to_visualize = generate_plot_ts(args, n_cols)
        # indices_to_visualize = indices_to_visualize if not reverse else list(reversed(indices_to_visualize))
        print("indices_to_visualize", indices_to_visualize)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

        for col, t_index in enumerate(indices_to_visualize):
            for row in range(n_rows):
                img = img_series[t_index, 0].numpy()
                # img = (img - img.min()) / (img.max() - img.min())
                if args.RGB_channel == 1:
                    axs[col].imshow(img, cmap="gray")
                else:
                    axs[col].imshow(img)
                axs[col].axis("off")
                if row == 0:
                    axs[col].set_title(f"t={t_index}")
        plt.tight_layout()
        plt.show()


def visualize_forward_diffusion(args, model, data):
    """
    Visualize the forward diffusion process at specific timesteps.

    Args:
        model: A PyTorch model that provides the forward_diffusion method.
        data: A PyTorch tensor or list of tensors containing the input data.
        args: An object or namespace that includes:
            - num_ts (int): The total number of timesteps.
            - device (str or torch.device): The device to use (e.g., 'cpu' or 'cuda').
    
    This function will:
        1. Define a list of specific timesteps.
        2. Generate forward-diffused samples at those timesteps.
        3. Plot the results in a grid where each row corresponds to a channel,
           and each column corresponds to one of the selected timesteps.
    """

    sample_ts = np.linspace(0, args.num_ts - 1, args.num_ts).astype(int)
    # Store results
    results = []

    # Ensure x0 has shape [1, C, H, W]
    x0 = torch.as_tensor(data[0].unsqueeze(0)).to(args.device)  
    print("x0 shape:", x0.shape)
    print("x0 range:", x0.min().item(), x0.max().item())

    # Forward diffusion
    with torch.no_grad():
        for t in sample_ts:
            t_tensor = torch.as_tensor([t], device=args.device)
            x_t, noise = model.forward_diffusion(x0, t_tensor)
            # Append the first sample (index 0) from the batch to the results
            results.append(x_t.cpu())

    print("Number of results:", len(results))

    results = torch.stack(results, dim=0)

    print("Results shape:", results.shape)
    # Plot the results
    plot_diffusion(args, results)
    return results

def visulize_reverse_diffusion(args, model, cond, exp_index=0):
    """
    Visualize the reverse diffusion process at specific timesteps.

    Args:
        model: A PyTorch model that provides the reverse_diffusion method.
        data: A PyTorch tensor or list of tensors containing the input data.
        args: An object or namespace that includes:
            - num_ts (int): The total number of timesteps.
            - device (str or torch.device): The device to use (e.g., 'cpu' or 'cuda').
    
    This function will:
        1. Define a list of specific timesteps.
        2. Generate reverse-diffused samples at those timesteps.
        3. Plot the results in a grid where each row corresponds to a channel,
           and each column corresponds to one of the selected timesteps.
    """
    condition = cond[exp_index].to(args.device)
    # generated_image, generated_x0 = model.sample(condition.unsqueeze(0), shape = (1, args.depth_channel, args.img_dim, args.img_dim, args.RGB_channel))
    generated_image = model.complex_sample(condition.unsqueeze(0), shape=(1, args.RGB_channel, args.depth_channel, args.img_dim, args.img_dim) if args.depth_channel > 1 else (1, args.RGB_channel, args.img_dim, args.img_dim)) # args.depth_channel,)c
    # Plot the results
    plot_diffusion(args, generated_image, reverse=True)

def visulize_ddim_sample(args, model, cond, exp_index=0, ddim_ts=11, n_cols=11):
    """
    Visualize the reverse diffusion process at specific timesteps.

    Args:
        model: A PyTorch model that provides the reverse_diffusion method.
        cond: A PyTorch tensor containing the conditioning input.
        args: An object or namespace that includes:
            - num_ts (int): The total number of timesteps.
            - device (str or torch.device): The device to use (e.g., 'cpu' or 'cuda').
    
    This function will:
        1. Define a list of specific timesteps.
        2. Generate reverse-diffused samples (x_t and x_0) at those timesteps.
        3. Plot the results in a grid where each row corresponds to a channel,
           and each column corresponds to one of the selected timesteps.
    """
    condition = cond[exp_index].to(args.device)
    print("condition:", condition)
    
    # Run DDIM sampling to get x_t and x_0
    x_t, x_0 = model.ddim_samcple(condition.unsqueeze(0), 
                                 shape=(1, args.RGB_channel, args.depth_channel, args.img_dim, args.img_dim) if args.depth_channel > 1 else (1, args.RGB_channel, args.img_dim, args.img_dim), # args.depth_channel,
                                 num_ts=ddim_ts,
                                 eta=0.0)
    
    x_t = x_t.flip(0)
    x_0 = x_0.flip(0)
    print("x_t shape:", x_t.shape)
    x_t = x_t.permute(0, 1, 3, 4, 5, 2) if x_t.ndim==6 else x_t.permute(0, 1, 3, 4, 2)# [T, 1, D, H, W, C] or [T, 1, H, W, C]
    x_0 = x_0.permute(0, 1, 3, 4, 5, 2) if x_0.ndim==6 else x_0.permute(0, 1, 3, 4, 2)# [T, 1, D, H, W, C] or [T, 1, H, W, C]
    
    indices_to_visualize = np.linspace(0, ddim_ts - 1, n_cols).astype(int)
    # indices_to_visualize = np.flip(indices_to_visualize, axis=0)
    print("indices_to_visualize", indices_to_visualize)

    n_rows = args.depth_channel if args.depth_channel >= 1 else 1
    fig, axs = plt.subplots(2 * n_rows, n_cols, figsize=(n_cols * 3, 2 * n_rows * 3))

    # Visualize x_t and x_0
    for col, t_index in enumerate(indices_to_visualize):
        for row in range(n_rows):  # Iterate over channels (rows)
            if x_t.ndim == 6:
                # Select x_t and x_0 images for the current timestep and row
                img_t = x_t[t_index, 0, row].numpy()
                img_0 = x_0[t_index, 0, row].numpy()
            elif x_t.ndim == 5:
                img_t = x_t[t_index, 0].numpy()
                img_0 = x_0[t_index, 0].numpy()

            # # Normalize x_t and x_0 for visualization
            # img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min())
            # img_0 = (img_0 - img_0.min()) / (img_0.max() - img_0.min())

            # Plot x_t
            axs[row, col].imshow(img_t, cmap="gray" if args.RGB_channel == 1 else None)
            axs[row, col].axis("off")
            if row == 0:
                axs[row, col].set_title(f"t={t_index} (x_t)", fontsize=20)

            # Plot x_0 (in the second row group)
            axs[row + n_rows, col].imshow(img_0, cmap="gray" if args.RGB_channel == 1 else None)
            axs[row + n_rows, col].axis("off")
            if row == 0:
                axs[row + n_rows, col].set_title(f"t={t_index} (x_0)", fontsize=20)
    
    plt.tight_layout()
    plt.show()
