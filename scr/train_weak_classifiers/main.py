import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import MyNN
from dataset import Dataset_Loader
from train_utils import ModelTrainer

def main(args):
    config_path = args.config
    # Load configuration
    with open(config_path, "r") as file:
        config = json.load(file)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"7"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = args.base_path
    data_path = config["data_path"]
    save_path = config["output"]["model_save_path"]
    tasks = args.tasks
    Z_Constant = args.Z_Constant
    multi_mixing_data_path = f"{base_path}/{data_path}/tasks_{tasks}/zipfian_constant_{Z_Constant}"
    models_save_path = f"{base_path}/{save_path}/tasks_{tasks}/zipfian_constant_{Z_Constant}/"
    # Load dataset
    X = np.load(f"{multi_mixing_data_path}/X.npy")
    y = np.load(f"{multi_mixing_data_path}/y.npy")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["data"]["test_size"], random_state=42)

    # Create datasets and loaders
    train_dataset = Dataset_Loader(X_train, y_train)
    test_dataset = Dataset_Loader(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    num_classes = len(np.unique(y_train))
    model = MyNN(num_classes, dropout_rate=config["model"]["dropout_rate"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train and evaluate
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        save_path=models_save_path,
        n_epochs=args.num_epochs,
        device='cuda',
        verbose=config["output"]["verbose"]
    )

    trainer.train_and_evaluate(num_models=args.num_models)
    print(f"Training completed. Models saved in {config['output']['model_save_path']}")    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a neural network.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--num_models", type=int, required=True)
    parser.add_argument("--Z_Constant", type=float, required=True)
    
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    
    args = parser.parse_args()
    main(args)
