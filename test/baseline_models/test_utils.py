import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import accuracy_score


def create_dataloader(data, labels, batch_size=32, shuffle=True):
    """
    Create a DataLoader from data and labels.

    Parameters:
        data: Input data as a NumPy array or tensor.
        labels: Corresponding labels as a NumPy array or tensor.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.

    Returns:
        A PyTorch DataLoader.
    """
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
def load_data(data_path, max_samples=50000):
    """
    Load .npy files from a specified directory and sample up to a maximum number of rows.

    Parameters:
        data_path: Path to the directory containing .npy files.
        max_samples: Maximum number of samples to load from each file (default: 50000).

    Returns:
        data_dict: Dictionary containing data loaded from .npy files.
    """
    data_dict = {}
    for file_name in os.listdir(data_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(data_path, file_name)
            data = np.load(file_path)

            # Sample the data if its length exceeds max_samples
            if data.shape[0] > max_samples:
                sampled_indices = np.random.choice(data.shape[0], max_samples, replace=False)
                data = data[sampled_indices]

            data_key = file_name.replace('.npy', '')  # Remove file extension for key
            data_dict[data_key] = data
            print(f"Loaded {data_key} with shape {data.shape}")
    return data_dict

def create_labels(data_dict):
    """
    Create labels for the dataset. Assign 0 to 'normal' data, 1 to all others.

    Parameters:
        data_dict: Dictionary with keys representing dataset names.

    Returns:
        labels_dict: Dictionary with the same keys as data_dict and labels as values.
    """
    labels_dict = {}
    for key in data_dict:
        if 'GOOSE_normal' in key:
            labels_dict[key] = 0
        else:
            labels_dict[key] = 1
    return labels_dict

import os
import pandas as pd
import os
import pandas as pd

def save_results_to_excel(results, save_path, file_name='results.xlsx', model_name=None):
    """
    Save the results to an Excel file in a format where attack names are column headings and models are rows.

    Parameters:
        results: Dictionary containing the results for each dataset (attack names as keys).
        save_path: Directory to save the Excel file.
        file_name: Name of the Excel file (default: 'results.xlsx').
        model_name: Name of the model to associate with the results (row identifier).
    """
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)

    # Convert results to a DataFrame (single row for the current model)
    results_df = pd.DataFrame([results], index=[model_name])

    if os.path.exists(file_path):
        # If the file exists, read the existing data
        existing_df = pd.read_excel(file_path, index_col=0)

        # Update existing data with new model results or add missing attacks
        combined_df = pd.concat([existing_df, results_df], axis=0)
    else:
        # If the file doesn't exist, use the new DataFrame
        combined_df = results_df

    # Save the DataFrame back to the Excel file
    combined_df.to_excel(file_path, index=True)
    print(f"Results saved to {file_path}")


def load_model(model_class, model_path, device='cuda'):
    """
    Load all PyTorch models from the specified directory.

    Parameters:
        model_class: The class of the model to be loaded (e.g., MyNN).
        model_path: Path to the directory containing saved model files (.pth).
        device: Device to load the models on ('cuda' or 'cpu').

    Returns:
        models: Dictionary where keys are model file names and values are loaded models.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Directory not found: {model_path}")

    models = []
    for file_name in os.listdir(model_path):
        if file_name.endswith('.pth'):
            full_path = os.path.join(model_path, file_name)
            # Initialize and load the model
            model = model_class()
            model.load_state_dict(torch.load(full_path, map_location=device))
            model.to(device)
            model.eval()  # Set the model to evaluation mode
            models.append(model)
            print(f"Model loaded successfully from {full_path}")

    if not models:
        print(f"No .pth models found in the directory: {model_path}")

    return models
def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on the test dataset.

    Parameters:
        model: The neural network model.
        test_loader: DataLoader for test data.
        criterion: Loss function.
        device: Device to evaluate on ('cuda' or 'cpu').

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs, _  = model(batch_data)
            predictions = torch.argmax(outputs, dim=1)
            binary_predictions = (predictions != 0).type(torch.long)
            correct += (binary_predictions == batch_labels).sum().item()
            total += batch_labels.size(0)
    accuracy = correct / total
    return accuracy


def evaluate_model_baseline(model, test_loader, device='cuda'):
    """
    Evaluate the model on the test dataset.

    Parameters:
        model: The neural network model.
        test_loader: DataLoader for test data.
        criterion: Loss function.
        device: Device to evaluate on ('cuda' or 'cpu').

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data).squeeze()

            # Get predictions
            preds = (outputs >= 0.5).float()  # Binary classification threshold
            preds = torch.atleast_1d(preds)
            # Collect predictions and targets
            if isinstance(preds, torch.Tensor):
                all_preds.extend(preds.cpu().tolist())
            else:
                all_preds.append(preds)  # Directly append the scalar value
            all_targets.extend(batch_labels.cpu().tolist())  # Move to CPU for accuracy calculation

    return accuracy_score(all_targets, all_preds)

def evaluate_ensemble(models, test_loader, device='cuda'):
    """
    Evaluate an ensemble of models on the test dataset using hard and soft ensembling.

    Parameters:
        models: List of neural network models.
        test_loader: DataLoader for test data.
        device: Device to evaluate on ('cuda' or 'cpu').

    Returns:
        Tuple of (hard ensemble accuracy, soft ensemble accuracy).
    """
    for model in models:
        model.eval()  # Set each model to evaluation mode

    hard_correct = 0
    soft_correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            # Collect outputs from all models
            outputs_list = [model(batch_data)[0] for model in models]
            # Soft Ensembling: Average the outputs (probabilities or logits) across models
            soft_outputs = torch.mean(torch.stack(outputs_list), dim=0)
            soft_predictions = torch.argmax(soft_outputs, dim=1)

            # Hard Ensembling: Majority vote on the predictions of each model
            hard_predictions = torch.mode(
                torch.stack([torch.argmax(output, dim=1) for output in outputs_list]), dim=0
            )[0]

            # Binary conversion (if necessary, e.g., 0 stays 0, others become 1)
            soft_binary_predictions = (soft_predictions != 0).type(torch.long)
            hard_binary_predictions = (hard_predictions != 0).type(torch.long)

            # Calculate correctness
            soft_correct += (soft_binary_predictions == batch_labels).sum().item()
            hard_correct += (hard_binary_predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    # Compute accuracy
    soft_ensemble_accuracy = soft_correct / total
    hard_ensemble_accuracy = hard_correct / total

    return hard_ensemble_accuracy, soft_ensemble_accuracy