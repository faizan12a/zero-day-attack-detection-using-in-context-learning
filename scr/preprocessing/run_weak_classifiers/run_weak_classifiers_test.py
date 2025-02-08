import os
import json
import argparse
import numpy as np
import torch
from run_weak_classifiers_utils import get_models_outputs_test

import os
import numpy as np

def process_and_save_data(input_dir, weakclassifierpath, batchsize,device,save_dir,classes):
    """
    Reads all .npy files from the input directory, processes them using the `get_models_outputs_rand` function,
    and saves the outputs to the specified save directory.

    Args:
        input_dir (str): Path to the directory containing .npy files.
        weakclassifierpath (str): Path to the weak classifier models.
        batchsize (int): Batch size for processing.
        save_dir (str): Path to the directory where output files will be saved.
    """
    # Ensure the save directory exists
    # os.makedirs(save_dir, exist_ok=True)
    
    # List all .npy files in the input directory
    npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    data_list = []
    # Process and overwrite .npy files
    for file in npy_files:
        file_path = os.path.join(input_dir, file)
        data = np.load(file_path)
        if data.shape[0] > 10000:  # Check if the file has more than 35,000 samples
            indices = np.random.choice(data.shape[0], 10000, replace=False)  # Randomly sample 35,000
            data = data[indices]
            np.save(file_path, data)
        data_list.append(data)
        
    # Process data and store outputs
    for i, (file_name, data) in enumerate(zip(npy_files, data_list)):
        print(f"Processing file: {file_name}...")
        print(data.shape)
        # Extract the name without the extension
        base_name = os.path.splitext(file_name)[0]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Call the provided function and include the original file name in the output
        output_path = os.path.join(save_dir, f"{base_name}_wc_output.npy")
        get_models_outputs_test(data, weakclassifierpath, batchsize, output_path,device,classes)

        print(f"Processed and saved output for {file_name} as {output_path}")

    print(f"All files processed and saved to {save_dir}.")

    
def main(args):
    # Load configuration
    config_path = args.config
    # Load configuration
    with open(config_path, "r") as file:
        config = json.load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = args.base_path
    data_path = config["data_path"]
    models_path = config["models_path"]
    save_path = config["save_path"]
    tasks = args.tasks
    Z_Constant = args.Z_Constant
    testing_data_path = f"{base_path}/{data_path}"
    models_save_path = f"{base_path}/{models_path}/tasks_{tasks}/zipfian_constant_{Z_Constant}"
    save_data_path = f"{base_path}/{save_path}tasks_{tasks}/zipfian_constant_{Z_Constant}"
    # Load dataset
    process_and_save_data(testing_data_path, models_save_path, 10000, device,save_data_path,tasks)

    # X = np.load(f"{multi_mixing_data_path}/X.npy")
    # get_models_outputs_rand(X,models_save_path,10000,multi_mixing_data_path,device,tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a neural network.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument('--tasks',type=int,required=True)
    parser.add_argument('--Z_Constant',type=float,required=True)
    parser.add_argument("--base_path", type=str, required=True)
    
    args = parser.parse_args()
    main(args)

