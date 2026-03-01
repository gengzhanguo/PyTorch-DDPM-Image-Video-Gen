import os
import time
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import trange
import wandb


class Trainer:
    """
    A class for managing the training and validation of a PyTorch model.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, args):
        """
        Initialize the Trainer.

        Args:
            model (torch.nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            args (Namespace): Configuration arguments.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.best_loss = float('inf')
        self.patience = 0

        # Create save directory if it doesn't exist
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        # Initialize wandb
        wandb.init(
            project=f"{self.args.dataset}_{self.args.unet_choice}_ddpm",
            config=self.args,
            name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )

    def train_one_epoch(self):
        """
        Train the model for one epoch and return the average training loss.
        """
        self.model.train()
        train_loss = 0.0

        for images, conditions in self.train_loader:
            loss = self.model.loss_function(images, conditions)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()

            # Uncomment to inspect gradient norms
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Layer: {name} | Grad Norm: {param.grad.norm().item()}")

            self.optimizer.step()

        self.scheduler.step()
        avg_loss = train_loss / len(self.train_loader)
        self.model.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """
        Validate the model and return the average validation loss.
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, conditions in self.val_loader:
                loss = self.model.loss_function(images, conditions)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        self.model.val_losses.append(avg_val_loss)

        return avg_val_loss

    def save_model(self):
        """
        Save the model's state dictionary to the specified directory.
        """
        model_path = os.path.join(
            self.args.save_dir,
            f"{self.args.dataset}_{self.args.unet_choice}_diffusion_model.pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print("Model saved!")

    def plot_losses(self):
        """
        Plot training and validation losses.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.model.train_losses, label='Train Loss')
        plt.plot(self.model.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


    def train(self):
        """
        Run the training loop for the specified number of epochs or until early stopping.
        """
        for epoch in trange(self.args.num_epochs):
            start_time = time.time()

            # Training phase
            avg_loss = self.train_one_epoch()

            # Validation phase
            avg_val_loss = self.validate()

            # Logging and printing
            print(f"Epoch {epoch + 1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            end_time = time.time()
            print(f"Epoch {epoch + 1} took: {end_time - start_time:.2f} seconds")

            wandb.log({"train_loss": avg_loss, "val_loss": avg_val_loss})

            # Save model if validation loss improves
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.patience = 0
                self.save_model()
            else:
                self.patience += 1
                if self.patience == self.args.max_patience:
                    print("Early stopping!")
                    break

            # Plot losses every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.plot_losses()

        # Final loss plot
        self.plot_losses()