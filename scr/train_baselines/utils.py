import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_save_path(base_path, save_path, model_name):
    """
    Creates a save path for the model. Goes into base_path/save_path, creates a folder
    with the model name, and saves the model as .pth. If a file with the same name exists,
    increments the index and saves as 'model_name_1.pth', 'model_name_2.pth', etc.

    Args:
        base_path (str): The base directory for saving models.
        save_path (str): The subdirectory within the base path.
        model_name (str): The name of the model.

    Returns:
        str: The final path where the model should be saved.
    """
    # Create the full directory path: base_path/save_path/model_name
    full_dir_path = os.path.join(base_path, save_path, model_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)
        print(f"Created directory: {full_dir_path}")

    # Create the model file path
    index = 0
    final_path = os.path.join(full_dir_path, f"{model_name}.pth")
    
    # Increment index if file with the same name exists
    while os.path.exists(final_path):
        index += 1
        final_path = os.path.join(full_dir_path, f"{model_name}_{index}.pth")

    return final_path

def save_model(model, save_path):
    """
    Saves the PyTorch model to the specified path.
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

def plot_loss(train_losses, save_path):
    """
    Plots and saves the loss curve.
    """
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig(save_path)
    print(f"Loss plot saved at {save_path}")
    plt.close()
    
def load_and_sample_data(dataset_folder, total_length, normal_ratio):
    """
    Loads datasets, samples a specific ratio of normal data, and balances remaining among attack datasets.

    Args:
        dataset_folder (str): Path to the folder containing all datasets.
        total_length (int): Total length of the training dataset to be sampled.
        normal_ratio (float): Ratio of normal samples in the dataset (0.0 to 1.0).

    Returns:
        torch.Tensor: Sampled data features.
        torch.Tensor: Corresponding labels.
    """
    # Identify dataset files
    exclude_keywords = ["randomreplay", "masqueradefakenormal", "masqueradefakefault", "poisonedhighrate"]
    dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith('.npy') and not any(keyword in f.lower() for keyword in exclude_keywords)]
    normal_file = [f for f in dataset_files if 'goose_normal' in f.lower()]
    attack_files = [f for f in dataset_files if 'goose_normal' not in f.lower()]

    if not normal_file:
        raise ValueError("No normal dataset file found in the folder.")

    # Load the normal dataset
    normal_data = np.load(os.path.join(dataset_folder, normal_file[0]))
    
    # Sample normal data
    normal_length = int(total_length * normal_ratio)
    if len(normal_data) < normal_length:
        raise ValueError(f"Not enough normal data: {len(normal_data)} samples available, {normal_length} required.")
    sampled_normal = normal_data[np.random.choice(len(normal_data), normal_length, replace=False)]
    normal_labels = np.zeros(normal_length)  # Label for normal data is 0

    # Load and sample attack data
    attack_length_per_file = int((total_length - normal_length) / len(attack_files))
    sampled_attack_data = []
    sampled_attack_labels = []

    for attack_file in attack_files:
        attack_data = np.load(os.path.join(dataset_folder, attack_file))
        if len(attack_data) < attack_length_per_file:
            raise ValueError(f"Not enough data in {attack_file}: {len(attack_data)} samples available, {attack_length_per_file} required.")
        sampled_attack = attack_data[np.random.choice(len(attack_data), attack_length_per_file, replace=False)]
        sampled_attack_data.append(sampled_attack)
        sampled_attack_labels.extend([1] * attack_length_per_file)  # Label for attack data is 1

    # Combine normal and attack data
    all_data = np.concatenate([sampled_normal] + sampled_attack_data, axis=0)
    all_labels = np.concatenate([normal_labels, sampled_attack_labels], axis=0)

    # Shuffle the combined dataset
    shuffle_indices = np.random.permutation(len(all_data))
    all_data = all_data[shuffle_indices]
    all_labels = all_labels[shuffle_indices]

    # Convert to PyTorch tensors
    return torch.tensor(all_data, dtype=torch.float32), torch.tensor(all_labels, dtype=torch.float32)
