import os
import json
import numpy as np
from utils_test import increase_classes_for_all_normalized, GPTTesting
from scipy.special import softmax
import argparse

def load_json_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def sample_goose_normal(Xtest_List, WC_test_List, headers, sample_size):
    """
    Sample a subset of data for inconsistent samples using GOOSE_normal data.

    Args:
        Xtest_List (list): List of all test data arrays.
        WC_test_List (list): List of all weak classifier outputs.
        headers (list): List of headers corresponding to test data.
        sample_size (int): Number of samples to select.

    Returns:
        tuple: Sampled X_InCon, WC_InCon, and y_InCon arrays.
    """
    # Find the index of GOOSE_normal in headers
    try:
        normal_index = next((i for i, header in enumerate(headers) if 'GOOSE_normal' in header), None)
    except ValueError:
        raise ValueError("GOOSE_normal not found in headers")

    # Extract normal data and weak classifier outputs
    normal_X = Xtest_List[normal_index]
    normal_WC = WC_test_List[normal_index]
    # Ensure consistent sampling between X and WC
    indices = np.random.choice(len(normal_X), size=sample_size, replace=False)
    X_InCon = normal_X[indices]
    WC_InCon = normal_WC[indices]
    y_InCon = np.zeros((sample_size,))  # Label as zeros for normal data
    return X_InCon, WC_InCon, y_InCon

# def load_test_data(test_data_path):
#     """Load test data from a directory and prepare headers, data, and labels."""
#     Xtest_List, ytest_List, headers = [], [], []
#     min = np.load("/home/mfaizan/programs/my_project/data/transformers/normalization/min.npy")
#     max = np.load("/home/mfaizan/programs/my_project/data/transformers/normalization/max.npy")
#     std = np.load("/home/mfaizan/programs/my_project/data/transformers/normalization/std.npy")
#     mean = np.load("/home/mfaizan/programs/my_project/data/transformers/normalization/mean.npy")
    
#     # X_train_ID[:,0:len(min)] = (X_train_ID[:,0:len(min)]-min)/(max-min)
#     for file in os.listdir(test_data_path):
#         if file.endswith('.npy'):
#             file_path = os.path.join(test_data_path, file)
#             X = np.load(file_path)
#             # X[:,0:len(min)] = (X[:,0:len(min)]-mean)/(std)
            
#             # X[:,0:len(min)] = (X[:,0:len(min)]-min)/(max-min)
#             header = os.path.splitext(file)[0]
#             headers.append(header)
#             label = 0 if 'GOOSE_normal' in header else 1
#             y = np.zeros((len(X),)) if label == 0 else np.ones((len(X),))
#             Xtest_List.append(X)
#             ytest_List.append(y)
#     return Xtest_List, ytest_List, headers


def load_test_data(test_data_path):
    """Load test data from a directory and prepare headers, data, and labels."""
    Xtest_List, ytest_List, headers = [], [], []
    min_vals = np.load("/home/mfaizan/programs/my_project/data/transformers/normalization/min.npy")
    max_vals = np.load("/home/mfaizan/programs/my_project/data/transformers/normalization/max.npy")
    std_vals = np.load("/home/mfaizan/programs/my_project/data/transformers/normalization/std.npy")
    mean_vals = np.load("/home/mfaizan/programs/my_project/data/transformers/normalization/mean.npy")
    
    allowed_files = {"GOOSE_masqueradefakenormal_test.npy", "GOOSE_randomreplay_test.npy", "GOOSE_normal_test.npy"}
    
    for file in os.listdir(test_data_path):
        if file in allowed_files:
            file_path = os.path.join(test_data_path, file)
            X = np.load(file_path)
            
            header = os.path.splitext(file)[0]
            headers.append(header)
            label = 0 if 'GOOSE_normal' in header else 1
            y = np.zeros((len(X),)) if label == 0 else np.ones((len(X),))
            Xtest_List.append(X)
            ytest_List.append(y)
    
    return Xtest_List, ytest_List, headers


def load_weak_classifiers(wc_path, headers, tasks, zipfian_constant):
    """Load weak classifier outputs from a directory."""
    WC_test_List = []
    for header in headers:
        wc_file = f"{header}_wc_output.npy"
        wc_file_path = os.path.join(wc_path, f"tasks_{tasks}", f"zipfian_constant_{zipfian_constant}", wc_file)
        if os.path.exists(wc_file_path):
            WC = np.load(wc_file_path)
            WC_test_List.append(WC)
        else:
            print(f"Warning: File not found {wc_file_path}")
    return WC_test_List

def prepare_wc_test_list(WC_test_List, tasks):
    """Prepare WC test list by applying softmax and increasing classes."""
    # WC_test_List_softmax = [WC for WC in WC_test_List]
    temperature = 2.5
    WC_test_List_softmax = [softmax(WC/temperature, axis=2) for WC in WC_test_List]
    return increase_classes_for_all_normalized(WC_test_List_softmax, tasks, 10**-5)

def load_transformers(transformer_path, tasks, zipfian_constant):
    """Load transformer-related parameters (future expansion)."""
    # Assuming the transformer path will require specific logic, placeholder for now.
    return os.path.join(transformer_path, f"tasks_{tasks}", f"zipfian_constant_{zipfian_constant}")
import os

def main():
    # Load configuration from JSON
    parser = argparse.ArgumentParser(description="Train and save models.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file.")
    parser.add_argument("--base_path", type=str, required=True)
    
    parser.add_argument("--wc_tasks", type=int, required=True)
    parser.add_argument("--wc_Z_Constant", type=float, required=True)
    
    parser.add_argument("--tf_tasks", type=int, required=True)
    parser.add_argument("--tf_Z_Constant", type=float, required=True)
    
    
    args = parser.parse_args()
    
    
    config = load_json_config(args.config)
    cuda_device = config["cuda_device"]
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_device}"
    model_name = config["model_name"]
    model_type = config["model_type"]
    base_path = args.base_path
    test_data_path = os.path.join(base_path, config['test_data_path'])
    wc_path = os.path.join(base_path, config['weak_classifier_path'])
    transformer_path = os.path.join(base_path, config['transformer_path'],model_name,model_type)
    performance_path_temp = config['performance_path']
    
    weak_classifier_tasks = args.wc_tasks
    weak_classifier_zipfian_constant = args.wc_Z_Constant
    
    transformer_tasks = args.tf_tasks
    transformer_zipfian_constant = args.tf_Z_Constant
    
    gpt_params = config['gpt_params']
    
    performance_path = f"{base_path}/{performance_path_temp}{model_name}/{model_type}/tasks_{transformer_tasks}/zipfian_constant_{transformer_zipfian_constant}/"
    # Load test data
    Xtest_List, ytest_List, headers = load_test_data(test_data_path)

    # Load weak classifiers
    WC_test_List = load_weak_classifiers(wc_path, headers, weak_classifier_tasks, weak_classifier_zipfian_constant)
    WC_test_List2 = prepare_wc_test_list(WC_test_List, transformer_tasks)
    X_InCon, WC_InCon, y_InCon = sample_goose_normal(Xtest_List, WC_test_List2, headers, sample_size=5000)

    # Placeholder for transformers logic
    transformers_path = load_transformers(transformer_path, transformer_tasks, transformer_zipfian_constant)

    input_dim = Xtest_List[0].shape[1]
    # Prepare GPT Testing
    testing = GPTTesting(
        model_directory=transformers_path,
        folder_path=performance_path,
        max_shots=gpt_params['max_Shots'],
        min_shots=gpt_params['min_Shots'],
        X_test_list=Xtest_List,
        y_test_list=ytest_List,
        X_InCon=X_InCon,
        y_InCon=y_InCon,
        mode=gpt_params['mode'],
        test_mode=gpt_params['test_mode'],
        tests_header=headers,
        testing_name=gpt_params['Testing_Name'],
        total_testing=gpt_params['Total_Testing'],
        WC_test_List=WC_test_List2,
        WC_InCon=WC_InCon,
        device=gpt_params['device'],
        INPUT_DIM=input_dim,
        Classes=transformer_tasks,
        NOS=gpt_params['num_of_samples'],
        NO_WC=gpt_params['NO_WC'],
        BATCH_SIZE=gpt_params['BATCH_SIZE'],
        model_name=model_name
    )

    # Run testing
    testing.run_testing()

if __name__ == "__main__":
    main()
