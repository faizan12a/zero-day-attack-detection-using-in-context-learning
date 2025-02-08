import os
import json
import argparse
import numpy as np
import torch
from run_weak_classifiers_utils import get_models_outputs_rand


def main(args):
    config_path = args.config
    # Load configuration
    with open(config_path, "r") as file:
        config = json.load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = args.base_path
    data_path = config["data_path"]
    models_path = config["models_path"]
    tasks = args.tasks
    Z_Constant = args.Z_Constant
    multi_mixing_data_path = f"{base_path}/{data_path}/tasks_{tasks}/zipfian_constant_{Z_Constant}"
    models_save_path = f"{base_path}/{models_path}/tasks_{tasks}/zipfian_constant_{Z_Constant}"
    # Load dataset
    X = np.load(f"{multi_mixing_data_path}/X.npy")
    get_models_outputs_rand(X,models_save_path,args.batch_size,multi_mixing_data_path,device,tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a neural network.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument('--tasks',type=int,required=True)
    parser.add_argument('--Z_Constant',type=float,required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument('--batch_size',type=int,required=True)
    
    args = parser.parse_args()
    main(args)
