import json
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from models import CNNModel, DNNModel, RNNModel, LSTMModel  # Import your models
from utils import create_save_path, save_model, plot_loss, load_and_sample_data
import argparse

def load_model(model_name, input_shape):
    """
    Loads the specified model based on the model_name.
    """
    if model_name == "CNN":
        return CNNModel(input_shape)
    elif model_name == "DNN":
        return DNNModel(input_shape)
    elif model_name == "RNN":
        return RNNModel(input_shape)
    elif model_name == "LSTM":
        return LSTMModel(input_shape)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

def train_model(model, dataloader, test_loader, criterion, optimizer, device = 'cuda', epochs=10):
    """
    Trains the given model using the provided DataLoader, criterion, and optimizer on the specified device.

    Args:
        model: The PyTorch model to train.
        dataloader: The DataLoader providing training data.
        criterion: The loss function.
        optimizer: The optimizer for training.
        device: The device to use for training (e.g., 'cuda' or 'cpu').
        epochs: Number of epochs to train the model.
    """
    model = model.to(device)  # Move the model to the specified device
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_loss = []
        
        for inputs, targets in dataloader:
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs.squeeze(), targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            epoch_loss.append(loss.item())
    

        # Average loss for the epoch
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {sum(epoch_loss)/len(epoch_loss):.4f}")
        train_accuracy = evaluate_model(model, dataloader)
        test_accuracy = evaluate_model(model, test_loader)
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    
    return train_losses

def evaluate_model(model, dataloader, device = 'cuda'):
    """
    Evaluates the model on the given DataLoader and computes accuracy on the specified device.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: The DataLoader providing evaluation data.
        device: The device to use for evaluation (e.g., 'cuda' or 'cpu').

    Returns:
        float: The accuracy score.
    """
    model = model.to(device)  # Move the model to the specified device
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()

            # Get predictions
            preds = (outputs >= 0.5).float()  # Binary classification threshold

            # Collect predictions and targets
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())  # Move to CPU for accuracy calculation

    return accuracy_score(all_targets, all_preds)


def main():
    # Load configuration from JSON
    parser = argparse.ArgumentParser(description="Run a Python script with a JSON config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load JSON configuration
    with open(args.config, "r") as config_file:
        config = json.load(config_file)
    
    model_name = config["model_name"]
    base_path = config["base_path"]
    save_path = config["save_path"]
    dataset_folder_temp = config["dataset_folder"]
    total_length = config["total_length"]
    normal_ratio = config["normal_ratio"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    dataset_folder = base_path+dataset_folder_temp
    # Load and sample datasets
    X, y = load_and_sample_data(dataset_folder, total_length, normal_ratio)
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Example input shape
    input_shape = (X_train.shape[1], X_train.shape[2]) if len(X_train.shape) > 2 else (X_train.shape[1],)
    if model_name == "CNN" or "LSTM" or "RNN":
        input_shape = (X_train.shape[1],1)
    # Load the model
    model = load_model(model_name, input_shape)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=epochs)

    # Evaluate the model
    

    # Save model and loss plot
    model_save_path = create_save_path(base_path, save_path, model_name.lower())
    save_model(model, model_save_path)
    plot_loss(train_losses, model_save_path.replace(".pth", "_loss.png"))

if __name__ == "__main__":
    main()
