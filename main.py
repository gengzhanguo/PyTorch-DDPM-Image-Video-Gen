import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
import sys
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import from src modules
from config import Config
from dataset import data_pipeline
from diffusion import DiffusionModel
from models.bottleneck_unet import Bottleneck_UNet
from models.earlyfusion_unet import EarlyFusion_UNet
from models.uniform_unet import Uniform_UNet
from models.openai_unet import UNetModel # Assuming UNetModel is the main class for openai_unet
from scripts.train import Trainer # Assuming Trainer class is in scripts/train.py
from utils.plot import plot_diffusion, visulize_ddim_sample, visualize_forward_diffusion, compare_beta_schedules

def get_unet_model(args):
    if args.unet_choice == "bottleneck":
        return Bottleneck_UNet(args, self_cond=args.self_cond, learn_var=args.learn_var)
    elif args.unet_choice == "earlyfusion":
        return EarlyFusion_UNet(args, self_cond=args.self_cond, learn_var=args.learn_var)
    elif args.unet_choice == "uniform":
        return Uniform_UNet(args, self_cond=args.self_cond, learn_var=args.learn_var)
    elif args.unet_choice == "openai":
        # Adjust parameters for OpenAI UNetModel if needed, based on args
        return UNetModel(
            in_channels=args.RGB_channel * args.depth_channel if args.dim_conv == 2 else args.RGB_channel,
            model_channels=args.base_channel,
            out_channels=args.RGB_channel * args.depth_channel if args.dim_conv == 2 else args.RGB_channel,
            num_res_blocks=args.base_mults[0] - 1, # Placeholder, adjust based on actual OpenAI UNet config
            attention_resolutions=[2, 4], # Placeholder, adjust
            dropout=0.1,
            channel_mult=args.base_mults,
            conv_resample=True,
            dims=args.dim_conv,
            num_classes=args.latent_dim, # Assuming latent_dim is used for class conditioning
            use_checkpoint=False,
            num_heads=args.num_head,
            use_scale_shift_norm=True # OpenAI UNet often uses this
        )
    else:
        raise ValueError(f"Unknown UNet choice: {args.unet_choice}")

def main():
    parser = argparse.ArgumentParser(description="DDPM for Image and Multi-frame Generation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample", "visualize"],
                        help="Operation mode: train, sample, or visualize.")
    parser.add_argument("--config_name", type=str, default="mnist",
                        help="Name of the config file (e.g., 'mnist', 'cifar100').")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to a model checkpoint for sampling or visualization.")
    
    # Add any other arguments that might be directly passed to Config or override it
    # These should largely match what\'s in src/config.py
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda, cpu, mps).")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset to use (MNIST, CIFAR100).")
    parser.add_argument("--depth_channel", type=int, default=1, help="Number of images per data point (for multi-frame).")
    parser.add_argument("--unet_choice", type=str, default="bottleneck",
                        choices=["earlyfusion", "uniform", "bottleneck", "openai", "test_unet"], help="UNet architecture choice.")
    parser.add_argument("--self_cond", action="store_true", help="Enable self-conditioning.")
    parser.add_argument("--learn_var", action="store_true", help="Enable learning variance.")

    parser.add_argument("--dataset_path", type=str, default="./dataset", help="path to data") # Corrected dataset_path

    cli_args, unknown = parser.parse_known_args()

    # Determine the config_name to load based on --dataset or --config_name
    resolved_config_name = cli_args.config_name
    if cli_args.dataset and cli_args.dataset.lower() != resolved_config_name.lower():
        # If dataset is explicitly provided and different from default config_name, use it
        resolved_config_name = cli_args.dataset.lower()

    # Load configuration using the resolved config_name
    args = Config(config_name=resolved_config_name) # Pass config_name directly

    # Manually apply CLI overrides to args.args if they are not None
    # This loop now overrides the values loaded from the YAML config
    for arg_name in ["device", "num_epochs", "batch_size", "lr", "dataset", "depth_channel", "unet_choice", "self_cond", "learn_var", "config_name", "dataset_path"]:
        cli_val = getattr(cli_args, arg_name)
        if cli_val is not None:
            setattr(args.args, arg_name, cli_val)

    # Manual validation for --dataset (still needed if Config doesn\'t validate it)
    valid_datasets = ["MNIST", "CIFAR100"]
    if args.args.dataset not in valid_datasets: # Check args.args.dataset which has been overridden
        parser.error(f"argument --dataset: invalid choice: '{args.args.dataset}' (choose from {', '.join(valid_datasets)})")

    # Re-evaluate image_channel based on potentially overridden depth_channel/RGB_channel
    if args.args.dim_conv == 2:
        args.args.image_channel = args.args.depth_channel * args.args.RGB_channel if args.args.depth_channel > 0 else args.args.RGB_channel
    elif args.args.dim_conv == 3:
        args.args.image_channel = args.args.RGB_channel
    
    # Initialize dataset and dataloaders
    print("Preparing data pipeline...")
    train_loader, val_loader, _, labels_raw = data_pipeline(args)
    
    # Initialize UNet model
    print(f"Initializing UNet model: {args.unet_choice}...")
    unet = get_unet_model(args).to(args.device)

    # Initialize Diffusion Model
    print("Initializing Diffusion Model...")
    model = DiffusionModel(args, unet).to(args.device)

    if cli_args.checkpoint_path:
        print(f"Loading checkpoint from {cli_args.checkpoint_path}")
        model.load_state_dict(torch.load(cli_args.checkpoint_path, map_location=args.device))
    
    if cli_args.mode == "train":
        print("Starting training mode...")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0) # Simple scheduler, can be customized
        trainer = Trainer(model, train_loader, val_loader, optimizer, scheduler, args)
        trainer.train()
    
    elif cli_args.mode == "sample":
        print("Starting sampling mode...")
        model.eval()
        with torch.no_grad():
            # Example: Sample one image of the first label
            # Need to get a valid condition for sampling
            condition_idx = 0
            if args.dataset == "MNIST":
                # For MNIST, condition is typically a single digit label
                # Assuming labels_raw contains original labels
                condition = labels_raw[condition_idx].unsqueeze(0).to(args.device)
                print(f"Sampling for condition: {labels_raw[condition_idx].item()}")
            elif args.dataset == "CIFAR100":
                # For CIFAR100, condition is typically a class label
                condition = labels_raw[condition_idx].unsqueeze(0).to(args.device)
                print(f"Sampling for condition: {labels_raw[condition_idx].item()}")
            else:
                condition = torch.randn(1, args.latent_dim).to(args.device) # Fallback for unconditioned

            sample_shape = (1, args.RGB_channel, args.depth_channel, args.img_dim, args.img_dim) if args.depth_channel > 1 else (1, args.RGB_channel, args.img_dim, args.img_dim)

            generated_image_series, generated_x0_series = model.sample(condition, sample_shape)
            
            # Save or display generated images
            save_dir = os.path.join(args.save_dir, "samples")
            os.makedirs(save_dir, exist_ok=True)
            
            # Plotting the last generated x0 image
            final_image = generated_x0_series[-1, 0, :, :, :] # Assuming last frame, first batch item
            if args.depth_channel > 1: # For video, might want to save all frames or a GIF
                # Placeholder for multi-frame saving/GIF creation
                print(f"Generated multi-frame output (depth_channel={args.depth_channel}). Consider saving as GIF.")
                # Example for saving the first frame if needed
                # save_image(final_image[0], os.path.join(save_dir, f"sample_final_{condition_idx}.png"))
            else:
                final_image_path = os.path.join(save_dir, f"sample_final_{condition_idx}.png")
                # Using plot_diffusion to save rather than torchvision.utils.save_image for consistency
                # plot_diffusion handles denormalization internally for display
                # For saving raw, need to handle denormalization explicitly if not already done
                from torchvision.utils import save_image
                save_image(final_image, final_image_path)
                print(f"Saved sample to {final_image_path}")

            # Optionally visualize the full diffusion process
            # plot_diffusion(args, generated_image_series, reverse=True, n_cols=min(11, args.num_ts))
            # plot_diffusion(args, generated_x0_series, reverse=True, n_cols=min(11, args.num_ts))
            
    elif cli_args.mode == "visualize":
        print("Starting visualization mode...")
        model.eval()
        with torch.no_grad():
            # This mode can be expanded to include various visualizations
            # Example: Visualize forward diffusion of a real image
            # For simplicity, we\'ll use a sample from the train_loader
            sample_data, _ = next(iter(train_loader))
            real_image = sample_data[0].unsqueeze(0).to(args.device) # Take first image from batch

            print("Visualizing forward diffusion of a real image...")
            forward_diffused_images = visualize_forward_diffusion(args, model, real_image)
            
            print("Visualizing DDIM reverse sampling process...")
            # For DDIM visualization, we need a condition
            condition_idx = 0
            condition = labels_raw[condition_idx].unsqueeze(0).to(args.device)
            visulize_ddim_sample(args, model, cond=labels_raw, exp_index=condition_idx, ddim_ts=args.num_ts, n_cols=min(11, args.num_ts))


if __name__ == "__main__":
    main()
