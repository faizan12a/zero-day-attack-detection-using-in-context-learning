import random
import numpy as np
from model import MyNN
import torch
import os

def get_models_outputs_rand(X_test, classifiers_path, batch_size):
    """
    Apply batches from X_test to randomly selected classifiers and get the probability outputs.
    
    Parameters:
    - X_test: Test features
    - classifiers_path: Path to the directory containing the classifier files
    - batch_size: The size of each batch to take from X_test

    Returns:
    - List of concatenated numpy arrays of probability outputs for each batch
    - Average inference time per sample
    """
    # Constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NO_WC = 10  # Number of classifiers to select randomly per batch
    Dim = X_test.shape[1]  # Input feature size

    # Get all classifier files
    all_classifiers = [f for f in os.listdir(classifiers_path) if os.path.isfile(os.path.join(classifiers_path, f))]
    if len(all_classifiers) < NO_WC:
        raise ValueError("Not enough classifiers in the directory to select from.")

    all_predictions = []  # To store predictions from all batches
    # Process X_test in batches
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size]
        batch_predictions = []
        
        # Randomly select 10 classifier files for each batch
        selected_classifiers_files = random.sample(all_classifiers, NO_WC)
        
        for classifier_file in selected_classifiers_files:
            file_path = os.path.join(classifiers_path, classifier_file)
            if classifier_file.endswith('.pth'):
                model = MyNN(Dim)  # Replace `MyNN` with your actual model class
                model.load_state_dict(torch.load(file_path, map_location=device))
                model.to(device)  # Move the model to the specified device
            
            model.eval()  # Set the model to evaluation mode
            batch = torch.Tensor(batch).to(device).float()
            probability_output, features = model(batch, Temp2=1.5)
            batch_predictions.append(probability_output.cpu().detach().numpy())
        
        # Concatenate the probability outputs along the first axis to keep them separate per model
        batch_predictions_array = np.array(batch_predictions).transpose(1, 0, 2)
        all_predictions.append(batch_predictions_array)
    
    # Combine all predictions
    all_predictions_array = np.concatenate(all_predictions, axis=0)
        
    return np.array(all_predictions_array)  