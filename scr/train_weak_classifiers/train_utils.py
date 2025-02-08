import os
import copy
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, loss_fn, save_path, n_epochs, device='cuda', verbose=True):
        """
        Initialize the ModelTrainer.

        Parameters:
            model: The PyTorch model to train.
            train_loader: DataLoader for training data.
            test_loader: DataLoader for testing data.
            loss_fn: Loss function.
            save_path: Path to save the trained models and loss plots.
            n_epochs: Number of training epochs.
            device: Device to train the model on ('cpu' or 'cuda').
            verbose: Whether to print detailed progress and log loss.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.n_epochs = n_epochs
        self.device = device
        self.verbose = verbose
        os.makedirs(save_path, exist_ok=True)

    def train_and_evaluate(self, num_models):
        """
        Train and evaluate multiple models.

        Parameters:
            num_models: Number of models to train.

        Returns:
            trained_models: List of trained models.
        """
        trained_models = []
        all_losses = []

        for model_idx in tqdm(range(num_models), desc="Training Models", leave=True):
            print(f"Training Model {model_idx + 1}/{num_models}")
            model_copy = copy.deepcopy(self.model).to(self.device)
            optimizer = optim.Adam(model_copy.parameters(), lr=0.001)
            epoch_loss = []
            epsilon=0.001
            for epoch in tqdm(range(self.n_epochs), desc=f"Model {model_idx + 1} Epochs", leave=False):
                model_copy.train()

                for data, target in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.n_epochs}", leave=False):
                    data, target = data.to(self.device), target.to(self.device)
                    data.requires_grad = True
                    
                    optimizer.zero_grad()
                    output = model_copy(data)
                    loss = self.loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
                    gradient = data.grad.data
                    data_perturbed = data + epsilon * gradient.sign()

                    # Forward pass with perturbed data
                    output_perturbed = model_copy(data_perturbed)
                    # output_perturbed = output_perturbed / temperature
                    loss_perturbed = self.loss_fn(output_perturbed, target)

                    optimizer.zero_grad()
                    loss_perturbed.backward()
                    optimizer.step()

                    epoch_loss.append(loss_perturbed.item())
                # # avg_epoch_loss = epoch_loss / len(self.train_loader)
                # epoch_losses.append(epoch_loss)

                # Evaluate on test data
                test_accuracy = self.evaluate_model(model_copy)

                if self.verbose:
                    print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss[-1]:.4f}, Test Accuracy: {test_accuracy:.4f}")

            # Save model and losses
            self.save_model(model_copy, model_idx + 1)
            self.plot_losses(epoch_loss, model_idx + 1)
            trained_models.append(model_copy)
            all_losses.append(epoch_loss)

    def evaluate_model(self, model):
        """
        Evaluate the model on the test dataset.

        Parameters:
            model: The trained model.

        Returns:
            accuracy: Test accuracy of the model.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                correct += (output.argmax(1) == target).sum().item()
                total += len(target)
        return correct / total

    def save_model(self, model, model_idx):
        """
        Save the model to the specified path. If a model with the same index already exists,
        increment the index until an unused filename is found.

        Parameters:
            model: The trained model.
            model_idx: Starting index of the model.
        """
        while True:
            model_save_path = os.path.join(self.save_path, f"model_{model_idx}.pth")
            if not os.path.exists(model_save_path):
                break
            model_idx += 1

        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")


    def plot_losses(self, losses, model_idx):
        """
        Plot and save training losses for a single model. If a plot with the same index
        already exists, increment the index until an unused filename is found.

        Parameters:
            losses: List of losses for each epoch.
            model_idx: Starting index of the model.
        """
        while True:
            plot_path = os.path.join(self.save_path, f"model_{model_idx}_loss.png")
            if not os.path.exists(plot_path):
                break
            model_idx += 1

        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(losses) + 1), losses, label=f"Model {model_idx}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss for Model {model_idx}")
        plt.legend()
        plt.grid()
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved to {plot_path}")


