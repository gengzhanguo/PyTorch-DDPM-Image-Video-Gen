#!/bin/bash

# This script provides examples for running the DDPM project (train, sample, visualize).
# Ensure you are in the DDPM_IMG/ directory before running this script.
# Usage: ./run.sh <command> [options]

# Set Python path to include the 'src' directory
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# --- Common Variables ---
# Adjust these as needed. Note: Most parameters are now expected to be passed directly via the command line.
DEVICE="cuda" # or "cpu", "mps"

# --- Functions for easier command execution ---
function train_model() {
    echo "Starting training..."
    python3 main.py \
        --mode train \
        --device "$DEVICE" \
        "$@"
}

function sample_images() {
    if [ -z "$1" ]; then
        echo "Error: Checkpoint path is required for sampling."
        echo "Usage: ./run.sh sample <path_to_checkpoint.pth> [additional_args]"
        return 1
    fi
    CHECKPOINT_PATH="$1"
    shift # Remove the first argument (checkpoint path)
    
    echo "Starting image sampling..."
    python3 main.py \
        --mode sample \
        --config_name "$CONFIG_NAME" \
        --device "$DEVICE" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --unet_choice "$UNET_CHOICE" \
        --depth_channel "$DEPTH_CHANNEL" \
        "$@"
}

function visualize_process() {
    if [ -z "$1" ]; then
        echo "Error: Checkpoint path is required for visualization."
        echo "Usage: ./run.sh visualize <path_to_checkpoint.pth> [additional_args]"
        return 1
    fi
    CHECKPOINT_PATH="$1"
    shift # Remove the first argument (checkpoint path)

    echo "Starting visualization..."
    python3 main.py \
        --mode visualize \
        --config_name "$CONFIG_NAME" \
        --device "$DEVICE" \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --unet_choice "$UNET_CHOICE" \
        --depth_channel "$DEPTH_CHANNEL" \
        "$@"
}

# --- Main execution logic ---
case "$1" in
    train)
        shift
        train_model "$@"
        ;;
    sample)
        shift
        sample_images "$@"
        ;;
    visualize)
        shift
        visualize_process "$@"
        ;;
    *)
        echo "Usage: ./run.sh {train|sample|visualize} [options]"
        echo "Examples:"
        echo "  ./run.sh train --num_epochs 200 --dataset CIFAR100 --unet_choice bottleneck"
        echo "  ./run.sh sample results/samples/my_model.pth --depth_channel 5" # Example with multi-frame
        echo "  ./run.sh visualize results/samples/my_model.pth"
        exit 1
        ;;
esac
