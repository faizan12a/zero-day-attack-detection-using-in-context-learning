import os
import json
import numpy as np
from test_utils import load_data, create_labels, save_results_to_excel, evaluate_model, create_dataloader,load_model, evaluate_ensemble, evaluate_model_baseline
from weak_classifier_model import MyNN
import torch.nn as nn
from baseline_models import CNNModel, DNNModel, RNNModel, LSTMModel  # Import your models
import argparse

def load_model_baseline(model_name):
    """
    Loads the specified model based on the model_name.
    """
    if model_name == "CNN":
        return CNNModel
    elif model_name == "DNN":
        return DNNModel
    elif model_name == "RNN":
        return RNNModel
    elif model_name == "LSTM":
        return LSTMModel
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
def main():
    # Load configuration from JSON file
    parser = argparse.ArgumentParser(description="Run a Python script with a JSON config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Load JSON configuration
    with open(args.config, "r") as config_file:
        config = json.load(config_file)

    # Extract paths and settings
    base_path = config['base_path']
    data_path_temp = config['data_path']
    results_save_path_temp = config['results_save_path']
    file_name = config.get('file_name', 'results.xlsx')  # Default to 'results.xlsx'
    model_name = config['model']
    tasks = config['tasks']
    batch_size = config['batch_size']
    model_path_temp = config['model_path']
    
    if "weak_classifier" in args.config: 
        zipfian_constant = config['zipfian_constant']
    data_path = base_path+data_path_temp
    results_save_path = base_path+results_save_path_temp
    model_path = base_path+model_path_temp

    # Step 1: Load data
    data_dict = load_data(data_path)
    first_key = next(iter(data_dict))  # Gets the first key in the dictionary
    input_dim = data_dict[first_key].shape  # Access shape[1] of the value
    # Step 2: Create labels
    labels_dict = create_labels(data_dict)
    device = "cuda"
    if model_name == "weak classifiers":
        model_path = f"{model_path}tasks_{tasks}/zipfian_constant_{zipfian_constant}"
        models = load_model(lambda: MyNN(n_classes=tasks), model_path, device)
        results_HE = {}
        results_SE = {}
        for dataset_name, data in data_dict.items():
            labels = np.full(data.shape[0], labels_dict[dataset_name])  # Create labels for the dataset
            test_loader = create_dataloader(data, labels, batch_size, shuffle=False)
            test_accuracy_HE, test_accuracy_SE = evaluate_ensemble(models, test_loader, device)
            
            results_HE[dataset_name] = test_accuracy_HE
            results_SE[dataset_name] = test_accuracy_SE

        # Step 4: Save results to Excel
        save_results_to_excel(results_HE, results_save_path, file_name,model_name=model_name+" Hard Ensembling")
        save_results_to_excel(results_SE, results_save_path, file_name,model_name=model_name+" Soft Ensembling")
        # Step 3: Evaluate models and collect results
    else:
        input_shape = (input_dim[1], input_dim[2]) if len(input_dim) > 2 else (input_dim[1],)
        if model_name == "CNN" or "LSTM" or "RNN":
            input_shape = (input_dim[1],1)
            print(input_shape)
        
        models = load_model(lambda: load_model_baseline(model_name)(input_shape = input_shape), model_path, device)
        model = models[-1]
        results = {}
        for dataset_name, data in data_dict.items():
            labels = np.full(data.shape[0], labels_dict[dataset_name])  # Create labels for the dataset
            test_loader = create_dataloader(data, labels, batch_size, shuffle=False)
            test_accuracy = evaluate_model_baseline(model, test_loader, device)
            
            results[dataset_name] = test_accuracy

        # Step 4: Save results to Excel
        save_results_to_excel(results, results_save_path, file_name,model_name=model_name)

if __name__ == '__main__':
    main()