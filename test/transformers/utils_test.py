import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from models import DTF,TF
import random

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
def plot_mean_std_with_error_bars(dataframe, save_path=None):
    # Assuming the first column contains x-values and the rest are y-values
    x_col = dataframe.columns[0]
    y_cols = dataframe.columns[1:]
    
    # Group the DataFrame by the x-values and compute mean and standard deviation for y-values
    grouped = dataframe.groupby(x_col)[y_cols].agg(['mean', 'std']).reset_index()

    # Extract x values and their means and standard deviations
    x_values = grouped[x_col]
    
    plt.figure(figsize=(10, 6))
    for y_col in y_cols:
        y_data = dataframe[y_col]
        y_mean = grouped[y_col]['mean']
        y_std = grouped[y_col]['std']
        plt.errorbar(x_values, y_mean, yerr=y_std, label=y_col)
    
    plt.ylim(0, 1)
    plt.xlim(-1, max(x_values))
    plt.yticks(np.arange(0, 1.1, 0.1))  # Set y ticks from 0 to 1 with a 0.1 interval
    plt.xticks(np.arange(-1,np.max(x_values),1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gcf().set_size_inches(6, 4)
    plt.xlabel(x_col)
    plt.ylabel("Accuracy")
    plt.title("Model")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    

class GPTTesting:
    def __init__(self, model_directory, folder_path, max_shots, min_shots, X_test_list, y_test_list, X_InCon, y_InCon, mode, test_mode, tests_header, testing_name, total_testing, WC_test_List, WC_InCon, device, INPUT_DIM, Classes, NOS, NO_WC, BATCH_SIZE,model_name):
        self.model_directory = model_directory
        self.folder_path = folder_path
        self.max_shots = max_shots
        self.min_shots = min_shots
        self.X_test_list = X_test_list
        self.y_test_list = y_test_list
        self.X_InCon = X_InCon
        self.y_InCon = y_InCon
        self.mode = mode
        self.test_mode = test_mode
        self.tests_header = tests_header
        self.testing_name = testing_name
        self.total_testing = total_testing
        self.WC_test_List = WC_test_List
        self.WC_InCon = WC_InCon
        self.device = device
        self.INPUT_DIM = INPUT_DIM
        self.Classes = Classes
        self.NOS = NOS
        self.NO_WC = NO_WC
        self.BATCH_SIZE = BATCH_SIZE
        self.model_files = [f for f in os.listdir(model_directory) if f.endswith('.pth')]
        self.headers = ['No of Shots'] + tests_header
        self.metrics_dict = {key: [] for key in self.headers}
        self.model_name = model_name

    def run_testing(self):
        os.makedirs(self.folder_path, exist_ok=True)

        with pd.ExcelWriter(os.path.join(self.folder_path, f"metrics_{self.testing_name}.xlsx")) as writer:
            for model_num, model_file in enumerate(self.model_files, start=1):
                model_path = os.path.join(self.model_directory, model_file)
                model = self.load_model2(model_path)
                model.to(self.device)
                
                shot_steps = self.calculate_shot_steps(self.min_shots, self.max_shots)
                shots = np.arange(self.min_shots, self.max_shots, shot_steps)

                for k in range(self.total_testing):
                    test_metrics = []
                    test_metrics = self.perform_testing(model, shots)
                    print(test_metrics)
                    self.save_metrics(test_metrics, writer, model_num, k)
                    self.metrics_dict = self.aggregate_metrics(self.metrics_dict, test_metrics)
                
            self.save_aggregated_metrics(writer)

    def load_model2(self, model_path):
        if self.model_name == "DTF":
            model = DTF(self.INPUT_DIM, self.Classes, self.NOS, self.NO_WC, self.device)
        elif self.model_name == "TF":
            model = TF(self.INPUT_DIM, self.Classes)
            
        # model.load_state_dict(torch.load("/home/mfaizan/programs/gpt2_powersystems_mod/data/models/Prob_tasks_300/model_test_np_0.958_Burstiness30_NOS_10_TotalSamples_120000_Epoch3_BS150_corrupt_normal_non_normal/Model_1.pth"))
        model.load_state_dict(torch.load(model_path))
        return model

    def calculate_shot_steps(self, min_shots, max_shots):
        if max_shots == 1:
            return 1
        step = round((max_shots - min_shots) * 0.1)
        return max(step, 1)

    def perform_testing(self, model, shots):
        test_metrics = {key: [] for key in self.headers}
        test_metrics[self.headers[0]].extend(shots.tolist())

        for X_test, y_test, test_header, WC_test in zip(self.X_test_list, self.y_test_list, self.headers[1:], self.WC_test_List):
            test_dataset = TestDatasetLoadOOD(X_test, self.X_InCon, y_test, self.y_InCon, WC_test, self.WC_InCon, 0, self.mode, self.test_mode, self.NOS, self.NO_WC, self.Classes,self.model_name)
            
            for num_shots in shots:
                test_dataset.add_shot(num_shots, self.NOS)
                dataloader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
                
                accuracy = self.evaluate_model(model, dataloader)
                print(f"Attack: {test_header}_Shot No: {num_shots}_Accuracy: {accuracy}")
                test_metrics[test_header].append(accuracy)
                
        return test_metrics

    def evaluate_model(self, model, dataloader):
        true_labels, predicted_labels = [], []
        
        
        with torch.no_grad():
            if self.model_name == "DTF":
                for data1, data2, target in dataloader:
                    data1, data2, target = data1.to(self.device), data2.to(self.device), target.squeeze().long().to(self.device)
                    output = model(data1, data2)
                    predicted = torch.argmax(output[:, -1], dim=1)
                    
                    test_label_binary = (target[:, -1] != 0).long()
                    predicted_binary = (predicted != 0).long()
                    true_labels.extend(test_label_binary.cpu().numpy())
                    predicted_labels.extend(predicted_binary.cpu().numpy())
                    # test_label = target[:, -1]
                    
                    # true_labels.extend(test_label.cpu().numpy())
                    # predicted_labels.extend(predicted.cpu().numpy())
            elif self.model_name == "TF":
                for data1, target in dataloader:
                    data1, target = data1.to(self.device), target.squeeze().long().to(self.device)
                    output = model(data1)
                    predicted = torch.argmax(output[:, -1], dim=1)
                    
                    test_label_binary = (target[:, -1] != 0).long()
                    predicted_binary = (predicted != 0).long()
                    true_labels.extend(test_label_binary.cpu().numpy())
                    predicted_labels.extend(predicted_binary.cpu().numpy())
                    
        return accuracy_score(true_labels, predicted_labels)

    def save_metrics(self, metrics, writer, model_num, test_num):
        df = pd.DataFrame(metrics)
        df.to_excel(writer, sheet_name=f"Model_{model_num}_WC_number_{test_num}_metrics", index=False)

    def aggregate_metrics(self, aggregated_metrics, new_metrics):
        for key in new_metrics:
            aggregated_metrics[key].extend(new_metrics[key])
        return aggregated_metrics

    def save_aggregated_metrics(self, writer):
        df = pd.DataFrame(self.metrics_dict)
        df.to_excel(writer, sheet_name="All_Models", index=False)
        plot_mean_std_with_error_bars(df, save_path=os.path.join(self.folder_path, f"plot_{self.testing_name}.png"))
        



def swap_matrix_and_array(matrix_prob, matrix_feat, array):
    # Ensure that all inputs are PyTorch tensors and are moved to the GPU
    matrix_prob = matrix_prob.to('cuda') if not matrix_prob.is_cuda else matrix_prob
    matrix_feat = matrix_feat.to('cuda') if not matrix_feat.is_cuda else matrix_feat
    array = array.to('cuda') if isinstance(array, torch.Tensor) and not array.is_cuda else torch.tensor(array).to('cuda')

    # Generate indices for matrix pairs and shuffle them
    num_pairs = matrix_feat.size(0)
    pair_indices = list(range(num_pairs))
    random.shuffle(pair_indices)

    # Create a new tensor for shuffled results
    shuffled_matrix_feat = torch.empty_like(matrix_feat, device='cuda')
    shuffled_matrix_prob = torch.empty_like(matrix_prob, device='cuda')
    shuffled_array = torch.empty_like(array, device='cuda')

    # Shuffle rows of the matrix_feat and matrix_prob tensors
    for i, pair_idx in enumerate(pair_indices):
        shuffled_matrix_feat[i] = matrix_feat[pair_idx]
        shuffled_matrix_prob[i] = matrix_prob[pair_idx]

    # Shuffle elements of the array according to the shuffled matrix pairs
    shuffled_array = array[pair_indices]

    return shuffled_matrix_prob, shuffled_matrix_feat, shuffled_array

class TestDatasetLoadOOD(Dataset):
    def __init__(self, X_test, X_InCon, y_test, y_InCon, WC_test, WC_InCon, num_shots, mode, test_mode, seq_len, NO_WC, Classes, model_name):
        # Perform train_test_split and prepare tensors on GPU
        X_test_split, X_OS, WC_test_split, WC_OS, y_test_split, y_OS = train_test_split(X_test, WC_test, y_test, test_size=0.3)
        print(X_test_split.shape)
        print(X_OS.shape)
        
        # if mode in [0, 1, 2] and test_mode == 0:
        y_IC = WC_InCon
        y_IC_OS = WC_OS
        y_IC_test = WC_test_split
        
        self.Classes, self.NO_WC = Classes, NO_WC
        self.model_name = model_name
        # Move data to GPU as torch tensors
        self.X_test_split = torch.tensor(X_test_split).float().to('cuda')
        self.X_InCon = torch.tensor(X_InCon).float().to('cuda')
        self.y_test_split = torch.tensor(y_test_split).float().to('cuda')
        self.y_InCon = torch.tensor(y_InCon).float().to('cuda')
        self.y_IC = torch.tensor(y_IC).float().to('cuda')
        self.X_OS = torch.tensor(X_OS).float().to('cuda')
        self.y_IC_OS = torch.tensor(y_IC_OS).float().to('cuda')
        self.y_OS = torch.tensor(y_OS).float().to('cuda')
        self.y_IC_test = torch.tensor(y_IC_test).float().to('cuda')
        self.num_of_samples = seq_len
        self.input_dim = X_test.shape[1]
        self.num_weak_classifiers = NO_WC
        # Process the data
        self.x_feat, self.x_prob, self.y = self.process_data(self.X_test_split, self.X_InCon, self.y_test_split, self.y_InCon, self.y_IC, self.X_OS, self.y_IC_OS, self.y_OS, self.y_IC_test, num_shots, seq_len)

        # Shots used in the process
        self.shots_x, self.shots_y = self.X_OS, self.y_IC_OS
        self.X_query, self.y_test, self.y_IC_test = self.X_test_split, self.y_test_split, self.y_IC_test

    def process_data(self, X_test_split, X_InCon, y_test_split, y_InCon, y_IC, X_OS, y_IC_OS, y_OS, y_IC_test, num_shots, seq_len):
        # Pre-allocate tensors on GPU
        processed_feat = torch.empty((X_test_split.size(0), seq_len + 1, X_test_split.size(1))).to('cuda')
        processed_prob = torch.empty((X_test_split.size(0), seq_len + 1, self.NO_WC, self.Classes)).to('cuda')
        processed_labels = torch.empty((y_test_split.size(0), seq_len + 1)).to('cuda')

        for i in range(len(X_test_split)):
            rand_ind_incon = torch.randint(0, len(X_InCon), (seq_len - num_shots,)).to('cuda')
            rand_ind_os = torch.randint(0, len(X_OS), (num_shots,)).to('cuda')

            # Collect features, probabilities, and labels for the current index using torch.cat
            feat = torch.cat((X_InCon[rand_ind_incon], X_OS[rand_ind_os]))
            prob = torch.cat((y_IC[rand_ind_incon], y_IC_OS[rand_ind_os]))
            labels = torch.cat((y_InCon[rand_ind_incon], y_OS[rand_ind_os]))
            # Swap them with your function (assumed to be GPU-compatible)
            prob, feat, labels = swap_matrix_and_array(prob, feat, labels)

            # Append the current test sample
            feat = torch.vstack((feat, X_test_split[i]))  # unsqueeze to match dimensions
            prob = torch.vstack((prob, y_IC_test[i].unsqueeze(0)))
            labels = torch.cat((labels, y_test_split[i].unsqueeze(0)))
            
            # Store the results in the processed arrays
            processed_feat[i] = feat  # clone() ensures values are copied
            processed_prob[i] = prob
            processed_labels[i] = labels
            # Clean up variables after each iteration
            del feat, prob, labels
            torch.cuda.empty_cache()
        # Convert all processed data to CPU and NumPy before returning
        return (
            processed_feat.cpu().numpy(),
            processed_prob.cpu().numpy(),
            processed_labels.cpu().numpy()
        )

    def add_shot(self, num_shots, seq_len):        
        self.x_feat, self.x_prob, self.y = self.process_data(self.X_query, self.X_InCon, self.y_test, self.y_InCon, self.y_IC, self.shots_x, self.shots_y, self.y_OS, self.y_IC_test, num_shots, seq_len)

    def __len__(self):
        return len(self.x_feat)

    def __getitem__(self, idx):
        # The data has been converted to NumPy arrays, so no need to move to GPU
        if self.model_name == "DTF":
            x_feat = torch.Tensor(self.x_feat[idx])
            x_prob = torch.Tensor(self.x_prob[idx])
            y = torch.Tensor(self.y[idx])

            return x_feat, x_prob, y
        elif self.model_name == "TF":
            x = torch.empty(2 * (self.num_of_samples+1), self.input_dim)
            padding_size = self.input_dim  - self.num_weak_classifiers
            padding = -1 * torch.ones((self.num_of_samples +1, padding_size), dtype=torch.float)
            x_labels_padded = torch.cat(( torch.Tensor(self.x_prob[idx]).argmax(dim=2), padding), dim=1)
            x[0::2, :] = torch.Tensor(self.x_feat[idx])
            x[1::2, :] = x_labels_padded
            return (x,
                    torch.Tensor(self.y[idx]))

# def swap_matrix_and_array(matrix_prob,matrix_feat, array):
#     # Check if matrix rows are even
#     # if len(matrix_feat) % 2 != 0:
#     #     raise ValueError("The number of rows in the matrix must be even.")
#     # if len(array) * 2 != len(matrix_feat):
#     #     raise ValueError("The length of the array must be half the number of rows in the matrix.")

#     # Generate indices for matrix pairs and shuffle them
#     num_pairs = len(matrix_feat)
#     pair_indices = list(range(num_pairs))
#     random.shuffle(pair_indices)

#     # Create a new copy of the matrix for shuffling
#     shuffled_matrix_feat = np.copy(matrix_feat)
#     shuffled_matrix_prob = np.copy(matrix_prob)
#     # Shuffle rows of the matrix in pairs
#     for i, pair_idx in enumerate(pair_indices):
#         original_pair_start = pair_idx  # Original position of the pair
#         new_pair_start = i  # New position of the pair
#         shuffled_matrix_feat[new_pair_start] = matrix_feat[original_pair_start]
#         shuffled_matrix_prob[new_pair_start] = matrix_prob[original_pair_start]

#     # Shuffle elements of the array according to the shuffled matrix pairs
#     shuffled_array = [array[idx] for idx in pair_indices]
#     return shuffled_matrix_prob.tolist(),shuffled_matrix_feat.tolist(), shuffled_array

# class TestDatasetLoadOOD(Dataset):
#     def __init__(self, X_test, X_InCon, y_test, y_InCon,WC_test,WC_InCon ,num_shots, mode, test_mode, seq_len,NO_WC,Classes):
#         X_test_split, X_OS, WC_test_split, WC_OS,y_test_split,y_OS = train_test_split(X_test, WC_test,y_test, test_size=0.3, random_state=45)
        
#         # if mode in [0, 1, 2] and test_mode == 0:
#         y_IC = WC_InCon
#         y_IC_OS = WC_OS
#         y_IC_test = WC_test_split
        
#         self.Classes, self.NO_WC = Classes,NO_WC
#         self.x_feat, self.x_prob, self.y = self.process_data(X_test_split, X_InCon, y_test_split, y_InCon, y_IC, X_OS, y_IC_OS, y_OS, y_IC_test, num_shots, seq_len)
#         self.shots_x, self.shots_y = X_OS, y_IC_OS
#         self.X_InCon, self.y_IC = X_InCon, y_IC
#         self.X_query, self.y_test, self.y_IC_test = X_test_split, y_test_split, y_IC_test
#         self.y_InCon, self.y_OS = y_InCon, y_OS
        
#     def process_data(self, X_test_split, X_InCon, y_test_split, y_InCon, y_IC, X_OS, y_IC_OS, y_OS, y_IC_test, num_shots, seq_len):
#         processed_feat = np.empty((len(X_test_split), seq_len + 1, X_test_split.shape[1]))
#         processed_prob = np.empty((len(X_test_split), seq_len + 1, self.NO_WC, self.Classes))
#         processed_labels = np.empty((len(y_test_split), seq_len + 1))

#         for i in range(len(X_test_split)):
#             rand_ind_incon = np.random.randint(0, len(X_InCon), size=(seq_len - num_shots))
#             rand_ind_os = np.random.randint(0, len(X_OS), size=num_shots)
            
#             feat, prob, labels = [], [], []
#             self.append_data(feat, prob, labels, X_InCon, y_IC, y_InCon, rand_ind_incon)
#             self.append_data(feat, prob, labels, X_OS, y_IC_OS, y_OS, rand_ind_os)

#             prob, feat, labels = swap_matrix_and_array(prob, feat, labels)
#             feat.append(X_test_split[i])
#             prob.append(y_IC_test[i])
#             labels.append(y_test_split[i])

#             processed_feat[i] = np.array(feat)
#             processed_prob[i] = np.array(prob)
#             processed_labels[i] = np.array(labels).reshape((len(labels)))

#         return processed_feat, processed_prob, processed_labels

#     def append_data(self, feat, prob, labels, X, y_IC, y, rand_indices):
#         for idx in rand_indices:
#             feat.append(X[idx])
#             prob.append(y_IC[idx])
#             labels.append(y[idx])

#     def add_shot(self, num_shots, seq_len):
#         self.x_feat, self.x_prob, self.y = self.process_data(self.X_query, self.X_InCon, self.y_test, self.y_InCon, self.y_IC, self.shots_x, self.shots_y, self.y_OS, self.y_IC_test, num_shots, seq_len)

#     def __len__(self):
#         return len(self.x_feat)

#     def __getitem__(self, idx):
#         x_feat = torch.Tensor(self.x_feat[idx])
#         x_prob = torch.Tensor(self.x_prob[idx])
#         y = torch.Tensor(self.y[idx])

#         return x_feat, x_prob, y
    
import numpy as np

def increase_classes_for_all_normalized(matrices, new_class_count, epsilon):
    """
    Increase the number of classes in the third dimension for each matrix in the list by adding
    positive values from a normal distribution with mean 0 and std deviation epsilon, and then
    normalize the values so that the probabilities across the third dimension sum to 1.

    Parameters:
        matrices (list of np.array): A list of 3D matrices with shape (samples, classifiers, classes).
        new_class_count (int): The desired number of total classes (new third dimension size).
        epsilon (float): The standard deviation for the normal distribution to add to extra classes.

    Returns:
        list of np.array: A list of new matrices with updated shape (samples, classifiers, new_class_count), normalized so the sum of probabilities across the third dimension is 1.
    """
    modified_matrices = []
    
    for matrix in matrices:
        samples, classifiers, original_class_count = matrix.shape
        
        # Check if we need to add more classes
        if new_class_count <= original_class_count:
            return matrices
            
        # Calculate how many new classes to add
        extra_classes = new_class_count - original_class_count
        
        # Generate positive values from a normal distribution with mean 0 and std epsilon
        extra_class_values = np.abs(np.random.normal(loc=0, scale=epsilon, size=(samples, classifiers, extra_classes)))
        
        # Concatenate the original matrix with the new class values along the third dimension
        new_matrix = np.concatenate([matrix, extra_class_values], axis=2)
        
        # Normalize so that the sum of probabilities for each sample (across classes) is 1
        new_matrix /= new_matrix.sum(axis=2, keepdims=True)
        
        # Add the modified matrix to the list
        modified_matrices.append(new_matrix)
    
    return modified_matrices