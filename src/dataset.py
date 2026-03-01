import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from collections import Counter
import random

from utils.helpers import normalize_data, normalize_label
# =========================== Dataset Processing ===========================
## ------------------------ MNIST Dataset Processing ---------------------------
class MNISTDatasetProcessor:
    def __init__(self, root, target_labels, depth_channel=3, pt_per_label=300):
        self.root = root
        self.target_labels = target_labels
        self.depth_channel = depth_channel
        self.pt_per_label = pt_per_label
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize to 32x32
            transforms.ToTensor()        # Convert to tensor
        ])
        self.dataset = datasets.MNIST(root=self.root, train=True, download=True)
        self.label_names = [str(i) for i in range(10)]

    def process(self):
        processed_data = []
        processed_labels = []

        for label in self.target_labels:
            label_data = [
                (self.transform(img).unsqueeze(0), lbl)  # Apply transform here
                for img, lbl in self.dataset if lbl == label
            ]

            random.shuffle(label_data)
            if len(label_data) < self.pt_per_label * self.depth_channel:
                raise ValueError(f"Not enough images for label {label}, found {len(label_data)} images.")

            # Select only required images

            if self.depth_channel > 0:
                label_data = label_data[:self.pt_per_label * self.depth_channel]
                images = torch.cat([img for img, _ in label_data], dim=0)  # Shape: (images_per_label, 1, 32, 32)
                images = images.unsqueeze(1)
                images = images.view(self.pt_per_label, 1, self.depth_channel, 32, 32)  # Shape: (100, 1, 3, 32, 32)
            else: 
                label_data = label_data[:self.pt_per_label]
                images = torch.cat([img for img, _ in label_data], dim=0)  # Shape: (images_per_label, 1, 32, 32)

            labels = torch.full((self.pt_per_label,), label)  # Shape: (100,)

            processed_data.append(images)
            processed_labels.append(labels)

        # Combine data and labels from all target labels
        processed_data = torch.cat(processed_data, dim=0)  # Shape: (num_batches, 1, 3, 32, 32)
        processed_labels = torch.cat(processed_labels, dim=0).unsqueeze(-1)  # Shape: (num_batches,)

        permutation= torch.randperm(processed_data.size(0))
        processed_data = processed_data[permutation]
        processed_labels = processed_labels[permutation]
        
        processed_data = normalize_data(processed_data)
        processed_labels = normalize_label(processed_labels)
        return processed_data, processed_labels


class MNISTCustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
## ------------------------- CIFAR100 Dataset Processing ---------------------------
class CIFAR100DatasetProcessor:
    def __init__(self, root, target_labels, depth_channel=3, pt_per_label=300):
        """
        :param root: Data set root directory
        :param target_labels: Target tag list
        :param depth_channel: The number of images per data point
        :param pt_per_label: Number of data points per label
        """
        self.root = root
        self.target_labels = target_labels
        self.depth_channel = depth_channel
        self.pt_per_label = pt_per_label
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  
            transforms.ToTensor(),      
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            # transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)) 
        ])
        self.dataset = datasets.CIFAR100(root=self.root, train=True, download=True)
        self.label_names = self.dataset.classes

    def process(self):
        processed_data = []
        processed_labels = []

        for label in self.target_labels:
            # Step 1: Collect all images with the same label
            label_data = [
                (self.transform(img), lbl)  # Apply transformations without adding batch dimension
                for img, lbl in self.dataset if lbl == label
            ]

            random.shuffle(label_data)

            # Calculate the maximum number of complete video sequences
            max_sequences = len(label_data) // self.depth_channel
            
            # If we don't have enough images for at least one full sequence, raise an error
            if max_sequences == 0:
                raise ValueError(f"Not enough images to form even one {self.depth_channel}-frame video for label {label}, found {len(label_data)} images.")
            
            # Trim label_data to only include images for complete sequences
            num_images_to_take = max_sequences * self.depth_channel
            label_data = label_data[:num_images_to_take]

            # Step 2: Extract images and stack them
            images = torch.stack([img for img, _ in label_data], dim=0)  # Shape: (num_images_to_take, channel, 32, 32)
            
            # Reshape to (num_sequences, depth_channel, channel, H, W) and then permute to (num_sequences, channel, depth_channel, H, W)
            images = images.view(max_sequences, self.depth_channel, images.shape[1], images.shape[2], images.shape[3])
            images = images.permute(0, 2, 1, 3, 4) # Final Shape: (num_sequences, channel, depth_channel, H, W)

            # Step 6: Create labels for the grouped data
            labels = torch.full((images.size(0),), label)  # Shape: (?,)

            # Append processed data and labels
            processed_data.append(images)
            processed_labels.append(labels)

        # Combine all data and labels
        processed_data = torch.cat(processed_data, dim=0)  # Shape: (total_batches, self.depth_channel, 32, 32, channel)
        processed_labels = torch.cat(processed_labels, dim=0).unsqueeze(-1)  # Shape: (total_batches, 1)

        permutation= torch.randperm(processed_data.size(0))
        processed_data = processed_data[permutation]
        processed_labels = processed_labels[permutation]

        # processed_data = normalize_data(processed_data)
        processed_labels = normalize_label(processed_labels)
        return processed_data, processed_labels


class CIFAR100CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        :param data: (num_batches, depth_channel, 3, 32, 32)
        :param labels: (num_batches, 1)
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ======================= Dataset Plotting =======================
def plot_dataset(args, data, labels, label_names, num_samples=10):
    print(1)
    if data.ndim == 5: # has the shape (num_batches, RGB_channel, depth_channel, 32, 32), with time dimension
        data = data.permute(0, 2, 3, 4, 1)  # Shape: (num_batches, depth_channel, 32, 32, RGB_channel)
        fig, axes = plt.subplots(num_samples, data.shape[1], figsize=(data.shape[1]*2, num_samples*2))
        if data.shape[1] == 1:
            for i in range(num_samples):
                sample = data[i][0].numpy()
                axes[i].imshow(sample)
                axes[i].axis('off')
                label = labels[i].item()
                index = int(np.round((label + 1) / 2 * (len(args.digits) - 1)))
                label = args.digits[index] 
                name = label_names[label] if label_names else label
                axes[i].set_title(f"Label: {label}-{name}")
        else:
            for i in range(num_samples):
                for j in range(data.shape[1]):
                    sample = data[i, j].numpy()
                    axes[i, j].imshow(sample)
                    axes[i, j].axis('off')
                    label = labels[i].item()
                    index = int(np.round((label + 1) / 2 * (len(args.digits) - 1)))
                    label = args.digits[index] 
                    name = label_names[label] if label_names else label
                    if j == 0:  # Only set title on the first column
                        axes[i, j].set_title(f"Label: {label}-{name}")
        plt.show()
    elif data.ndim == 4: # has the shape (num_batches, 32, 32, channel), without time dimension
        data = data.permute(0, 2, 3, 1)  # Shape: (num_batches, 32, 32, channel)
        fig, axes = plt.subplots(num_samples, 1, figsize=(2, num_samples*2))
        for i in range(num_samples):
            sample = data[i].numpy()
            axes[i].imshow(sample)
            axes[i].axis('off')
            label = labels[i].item()
            index = int(np.round((label + 1) / 2 * (len(args.digits) - 1)))
            label = args.digits[index] 
            name = label_names[label] if label_names else label
            axes[i].set_title(f"Label: {label}-{name}")
        plt.show()

# =========================== Training and Validation Splitting ===========================
def split_train_val(args, train_loader: DataLoader) -> Tuple[DataLoader, DataLoader]:
    """
    Dynamically partition training and validation DataLoader
    Parameters:
    - train_loader: indicates the complete training data loader.
    - train_ratio: training set ratio. For example, 0.8 indicates that 80% is used for training and 20% is used for verification.
    - batch_size: specifies the size of each batch.

    return:
    - train_sub_loader: indicates the partitioned training set DataLoader.
    - val_loader: specifies the partitionalized verification set DataLoader.
    """
    train_size = len(train_loader.dataset)
    indices = np.random.permutation(train_size)
    
    split = int(np.floor(args.split_ratio * train_size))
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_sub_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_loader.dataset, batch_size=args.batch_size, sampler=val_sampler)
    
    return train_sub_loader, val_loader

# =========================== Dataset Loading ===========================
def data_pipeline(args):
    """
    :param args: Configuration parameter object, which must be included: dataset_path, digits, depth_channel, pt_per_digit, batch_size 
    :param args.dataset: ('CIFAR100' or 'MNIST')
    :return: train_loader, val_loader, processed_data, processed_labels
    """
    if args.dataset == 'CIFAR100':
        processor = CIFAR100DatasetProcessor(
            root=args.dataset_path,
            target_labels=args.digits,
            depth_channel=args.depth_channel,
            pt_per_label=args.pt_per_digit
        )
    elif args.dataset == 'MNIST':
        processor = MNISTDatasetProcessor(
            root=args.dataset_path,
            target_labels=args.digits,
            depth_channel=args.depth_channel,
            pt_per_label=args.pt_per_digit
        )
    else:
        raise ValueError("Unsupported dataset type. Choose 'CIFAR100' or 'MNIST'.")

    processed_data, processed_labels = processor.process()

    if args.dataset == 'CIFAR100':
        dataset = CIFAR100CustomDataset(processed_data, processed_labels)
    else:
        dataset = MNISTCustomDataset(processed_data, processed_labels)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Processed Data Shape: {processed_data.shape}")
    print(f"Processed Labels Shape: {processed_labels.shape}")

    label_names = processor.label_names
    print(f"Label Names: {label_names}")

    for i, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        plot_dataset(args, x, y, label_names, num_samples=4)
        break
    
    for data, label in dataloader:
        print(f"Batch Data Shape: {data.shape}")
        print(f"Batch Labels Shape: {label.shape}")
        break

    train_loader, val_loader = split_train_val(args, dataloader)

    for images, labels in train_loader:
        args.dataset_shape = images.shape
        print(f"Train Batch Shape: {images.shape}, {labels.shape}")
        print('Image pixel range:', images.min(), images.max())
        print('Label range:', labels.min(), labels.max())
    
        break

    return train_loader, val_loader, processed_data, processed_labels
