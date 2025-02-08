import argparse
import json
import numpy as np
import torch
from utils import groundtruth_replacement, corrupt_non_normal_probabilities, OODDataset, corrupt_probabilities, ModelTrainer
from scipy.special import softmax
import os


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and save models.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file.")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--Z_Constant", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--num_models", type=int, required=True)
    
    args = parser.parse_args()


    # Read hyperparameters from the JSON file
    with open(args.config, 'r') as file:
        config = json.load(file)
    tasks = args.tasks
    num_models = args.num_models
    Z_Constant = args.Z_Constant
    base_path = args.base_path
    epochs = args.num_epochs
    batch_size = args.batch_size
    # Set up environment
    cuda_device = config["cuda_device"]
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_device}"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Extract subsystems from config
    data = config["data"]
    dataset = config["dataset"]
    training = config["training"]
    model_params = config["model"]

    model_name = model_params["model_name"]
    model_type = model_params["model_type"]
    
    # Build paths and hyperparameters
    data_folder = f"{args.base_path}/{data['TF_Data']}tasks_{tasks}/zipfian_constant_{Z_Constant}/"

    
    X_train_ID = np.load(data_folder + "X.npy")
    # X_train_ID[:, 0:len(min)] = (X_train_ID[:, 0:len(min)] - mean) / (std)

    y_train_ID = np.load(data_folder + "y.npy")
    WC_X_train = np.load(data_folder + "wc_output.npy", mmap_mode='r')
    # temperature = 2  # Adjust this value as needed
    # WC_X_train = softmax(WC_X_train / temperature, axis=2)  # Apply softmax with temperature scaling

    # Assuming WC_X_train is already defined
    num_samples = WC_X_train.shape[0]

    # Generate an array of temperatures, one for each sample, chosen randomly from [1, 1.5, 2, 2.5, 3]
    temperature_array = np.random.choice([1, 1.5, 2, 2.5, 3], size=num_samples)

    # Reshape temperature_array to match the dimensions of WC_X_train for broadcasting
    temperature_array_reshaped = temperature_array[:, np.newaxis, np.newaxis]

    # Apply temperature scaling and softmax
    scaled_WC_X_train = softmax(WC_X_train / temperature_array_reshaped, axis=2)

    WC_X_train = scaled_WC_X_train
    Classes = len(np.unique(y_train_ID))
    input_dim = X_train_ID.shape[1]

    # Process data
    normal_indexes = np.where((y_train_ID == 0))[0]
    WC_X_train_writable = WC_X_train.copy()
    
    if model_type == "Mix":
        if model_name == "DTF":
            WC_X_train_writable = groundtruth_replacement(WC_X_train_writable, y_train_ID, perc=0.05)
        elif model_name == "TF":
            WC_X_train_writable = groundtruth_replacement(WC_X_train_writable, y_train_ID, perc=0.6)
    
    non_normal_WC = WC_X_train_writable[np.where((y_train_ID != 0))[0]]
    if model_name == "DTF":
        corrupt_non_normal_WC = corrupt_non_normal_probabilities(non_normal_WC, 0.4)
    elif model_name == "TF":
        corrupt_non_normal_WC = corrupt_non_normal_probabilities(non_normal_WC, 0.4)

    normal_WC = WC_X_train[normal_indexes]
    if model_name == "DTF":
        corrupt_normal_WC = corrupt_probabilities(normal_WC, Classes, 0.05)
    elif model_name == "TF":
        corrupt_normal_WC = corrupt_probabilities(normal_WC, Classes, 0.01)
        

    WC_X_train_writable[normal_indexes] = corrupt_normal_WC
    WC_X_train_writable[np.where((y_train_ID != 0))] = corrupt_non_normal_WC

    # visualize_exact_correct_predictions(WC_X_train_writable[np.where((y_train_ID != 0))],y_train_ID[np.where((y_train_ID != 0))[0]])
    # visualize_exact_correct_predictions(corrupt_non_normal_WC,y_train_ID[np.where((y_train_ID != 0))[0]])
    # visualize_exact_correct_predictions(WC_X_train_writable[normal_indexes] ,y_train_ID[np.where((y_train_ID == 0))[0]])
    # visualize_exact_correct_predictions(corrupt_normal_WC,y_train_ID[np.where((y_train_ID == 0))[0]])

    print("Processing data completed")

    # Create dataset
    ID_samples = round((1 - dataset["Perc_OOD"]) * dataset["Samples"])
    P_OOD_samples = round(dataset["Perc_OOD"] * dataset["Samples"])
    final_dataset = OODDataset(
        X_train_ID, y_train_ID, WC_X_train_writable, 0, ID_samples,
        P_OOD_samples, dataset["num_of_samples"], Classes, dataset["num_weak_classifiers"],model_name
    )

    print("Dataset created")

    
    # Model and training
    model_save_path = f"{base_path}/{data['TF_Save_Path']}tasks_{tasks}/zipfian_constant_{Z_Constant}/"
    print(model_save_path)
    print("Training models...")
    TF_Model = ModelTrainer(
        model_save_path, num_models, epochs, batch_size,
        final_dataset, input_dim, Classes, dataset["num_of_samples"],
        training["NO_WC"], device, model_name = model_name
    )
    TF_Model.train_and_save_models()


if __name__ == "__main__":
    main()
