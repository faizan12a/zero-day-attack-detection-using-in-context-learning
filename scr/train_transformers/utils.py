import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from transformers import get_scheduler
import numpy as np
from models import DTF,TF

import numpy as np
import torch
from torch.utils.data import Dataset

def modify_classifier_output(probs, ground_truth, epsilon=1e-6):
    """
    Modify a classifier's probability output to peak at the ground truth label 
    with small random values at remaining classes.

    Parameters:
    probs (np.ndarray): 1D array of probabilities for each class.
    ground_truth (int): The ground truth label index.
    epsilon (float): The upper limit for small random values for non-ground truth classes.

    Returns:
    np.ndarray: Modified probability output.
    """
    new_value_0 = np.random.uniform(0.4, 0.8)
    
    # Calculate the total remaining probability
    remaining_prob = 1 - new_value_0
    ground_truth2 = int(ground_truth)
    
    # Adjust the rest of the probabilities proportionally
    non_gt_indices = np.arange(len(probs)) != ground_truth2
    probs[non_gt_indices] = probs[non_gt_indices] * (remaining_prob / np.sum(probs[non_gt_indices]))
    
    # Set the new value for the 0th index
    probs[ground_truth2] = new_value_0
    
    return probs

def groundtruth_replacement(prob_row,y,perc):
    indexes = select_random_index_tuples(prob_row, percentage = perc)
    for index in indexes:
        prob_row[index[0]][index[1]] = modify_classifier_output(prob_row[index[0]][index[1]],y[index[0]])
    return prob_row

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



class OODDataset(Dataset):
    def __init__(self, X_train, y_train, WC_X_train, mode, ID_samples, P_OOD_samples, num_of_samples, num_classes, num_weak_classifiers,model_name):
        self.num_classes = num_classes
        self.model_name = model_name
        # Move the relevant arrays to GPU (CUDA)
        X_train = torch.tensor(X_train).to('cuda')
        y_train = torch.tensor(y_train).to('cuda')
        WC_X_train_temp = torch.tensor(WC_X_train).to('cuda')
        self.input_dim = X_train.shape[1]
        if mode == 0:
            y_IC = WC_X_train_temp.clone()
        self.num_weak_classifiers = num_weak_classifiers
        self.num_of_samples = num_of_samples
        # Keep these on the CPU as NumPy arrays
        total_samples = 20 * round(ID_samples / 20) + P_OOD_samples
        self.features = np.empty((total_samples, num_of_samples + 1, X_train.shape[1]))
        self.probabilities = np.empty((total_samples, num_of_samples + 1, num_weak_classifiers, num_classes))
        self.labels = np.empty((total_samples, num_of_samples + 1))

        self._create_processed_data(X_train, y_train, y_IC, total_samples, num_of_samples, ID_samples)

        # Clear memory for large objects no longer needed
        del X_train, y_train, WC_X_train_temp, y_IC
        torch.cuda.empty_cache()  # Clear unused GPU memory

    def _create_processed_data(self, X_train, y_train, y_IC, total_samples, num_of_samples, ID_samples):
        rand_samples = int(20 * ID_samples / 20)

        for i in range(rand_samples):
            # if i % 10000 == 0 or i>=70000:
            #     print(i)
            self._generate_samples(X_train, y_train, y_IC, i, num_of_samples)

        # Find normal and attack indices
        normal_indexes = torch.where(y_train == 0)[0]
        attack_indexes = torch.where(y_train != 0)[0]
        X_normal_feat, X_normal_prob, y_normal = X_train[normal_indexes], y_IC[normal_indexes], y_train[normal_indexes]
        X_attack_feat, X_attack_prob, y_attack = X_train[attack_indexes], y_IC[attack_indexes], y_train[attack_indexes]

        A_N_samples = int(0 * ID_samples / 20)
        for i in range(rand_samples, A_N_samples + rand_samples):
            sequence = torch.randint(0, num_of_samples // 3, (1,)).item()
            rand_ind_normal = torch.randint(0, len(X_normal_feat), (num_of_samples - sequence,)).to('cuda')
            rand_ind_attack = torch.randint(0, len(X_attack_feat), (sequence,)).to('cuda')
            

            if i % 2 == 0:
                rand_ind_query = torch.randint(0, len(X_attack_feat),(1,)).item()
                self._process_indices(X_normal_feat, X_normal_prob, y_normal, X_attack_feat, X_attack_prob, y_attack, rand_ind_normal, rand_ind_attack,rand_ind_query, i)
            else:
                rand_ind_query = torch.randint(0, len(rand_ind_normal),(1,)).item()
                self._process_indices(X_attack_feat, X_attack_prob, y_attack, X_normal_feat, X_normal_prob, y_normal, rand_ind_attack, rand_ind_normal,rand_ind_query, i)

            # Free memory after each loop iteration
            del rand_ind_normal, rand_ind_attack
            torch.cuda.empty_cache()

        for i in range(A_N_samples + rand_samples, total_samples):
            num_shots = torch.randint(num_of_samples // 4, num_of_samples+1, (1,)).item()
            rand_ind = torch.randint(0, len(X_train), (num_of_samples - num_shots,)).to('cuda')
            rand_ind4 = torch.randint(0, len(y_train), (1,)).item()

            all_indexes = torch.where(y_train == y_train[rand_ind4])[0]
            X_new_feat, X_new_prob, y_new = X_train[all_indexes], y_IC[all_indexes], y_train[all_indexes]
            X_new_feat, X_new_prob, y_new = X_new_feat.to('cuda'), X_new_prob.to('cuda'), y_new.to('cuda')
            rand_ind2 = torch.randint(0, len(X_new_feat), (num_shots,)).to('cuda')
            rand_ind3 = torch.randint(0, len(all_indexes), (1,)).item()

            self._process_indices(X_train, y_IC, y_train, X_new_feat, X_new_prob, y_new, rand_ind, rand_ind2, rand_ind3, i)

            # Free memory after each loop iteration
            del rand_ind, rand_ind2, X_new_feat, X_new_prob, y_new, all_indexes
            torch.cuda.empty_cache()

    def _generate_samples(self, X_train, y_train, y_IC, i, num_of_samples):
        rand_ind = torch.randint(0, len(X_train), (num_of_samples + 1,)).to('cuda')
        features = X_train[rand_ind]
        probabilities = y_IC[rand_ind]
        labels = y_train[rand_ind]

        # Perform any required swapping or manipulation here (on GPU)
        probabilities, features, labels = swap_matrix_and_array(probabilities, features, labels)

        # Move back to CPU and store in numpy arrays
        self.features[i] = features.cpu().numpy()
        self.probabilities[i] = probabilities.cpu().numpy()
        self.labels[i] = labels.cpu().numpy()

        # Free memory after each iteration
        del rand_ind, features, probabilities, labels
        torch.cuda.empty_cache()

    def _process_indices(self, X_train, y_IC, y_train, X_new_feat, X_new_prob, y_new, rand_ind, rand_ind2, rand_ind3, i):
        features = torch.cat((X_train[rand_ind], X_new_feat[rand_ind2])).to('cuda')
        probabilities = torch.cat((y_IC[rand_ind], X_new_prob[rand_ind2])).to('cuda')
        labels = torch.cat((y_train[rand_ind], y_new[rand_ind2])).to('cuda')

        query_feature = X_new_feat[rand_ind3].to('cuda')
        query_prob = X_new_prob[rand_ind3].to('cuda')
        query_label = y_new[rand_ind3].to('cuda')

        probabilities, features, labels = swap_matrix_and_array(probabilities, features, labels)

        query_prob = query_prob.unsqueeze(0).to('cuda')  # Adding an additional dimension
        query_label = query_label.unsqueeze(0).to('cuda')  # Adding an additional dimension
        features = torch.vstack([features, query_feature]).to('cuda')
        probabilities = torch.vstack([probabilities, query_prob]).to('cuda')
        labels = torch.cat([labels, query_label]).to('cuda')

        # Move back to CPU and store in numpy arrays
        self.features[i] = features.cpu().numpy()
        self.probabilities[i] = probabilities.cpu().numpy()
        self.labels[i] = labels.cpu().numpy()

        # Free memory after each iteration
        del features, probabilities, labels, query_feature, query_prob, query_label
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.model_name == "DTF":
            return (torch.Tensor(self.features[idx]),
                    torch.Tensor(self.probabilities[idx]),
                    torch.Tensor(self.labels[idx]))
        elif self.model_name == "TF":
            
            x = torch.empty(2 * (self.num_of_samples+1), self.input_dim)
            padding_size = self.input_dim  - self.num_weak_classifiers
            padding = -1 * torch.ones((self.num_of_samples +1, padding_size), dtype=torch.float)
            x_labels_padded = torch.cat(( torch.Tensor(self.probabilities[idx]).argmax(dim=2), padding), dim=1)
            x[0::2, :] = torch.Tensor(self.features[idx])
            x[1::2, :] = x_labels_padded
            return (x,
                    torch.Tensor(self.labels[idx]))
def distribute_value(value):
    """
    Distributes a value into two smaller values where the first value is greater than the second.
    
    Parameters:
    - value: The value to be distributed.
    
    Returns:
    - A tuple containing two smaller values such that the first is greater than the second.
    """
    # Ensure that value1 is greater than value2
    value1 = random.uniform(value / 2, value)  # value1 is at least half of the value
    value2 = value - value1  # value2 is the remainder
    
    return value1, value2



def select_random_index_tuples(array, percentage=0.15):
    """
    Selects random index tuples across the first two dimensions of a 3D array.

    Parameters:
    - array: 3D numpy array
    - percentage: float, percentage of elements to select

    Returns:
    - selected_index_tuples: list of tuples, where each tuple is (index_1, index_2)
    """
    # Get the shape of the array
    dim1, dim2, _ = array.shape

    # Calculate the total number of elements in the first two dimensions
    total_elements = dim1 * dim2

    # Calculate the number of elements to select
    num_samples = int(percentage * total_elements)

    # Create a flattened index array for the first two dimensions
    flattened_indices = np.arange(total_elements)

    # Randomly select indices without replacement
    random_indices = np.random.choice(flattened_indices, num_samples, replace=False)

    # Convert the flattened indices back to 2D indices (for the first two dimensions)
    selected_2d_indices = np.unravel_index(random_indices, (dim1, dim2))

    # Combine the indices into tuples
    selected_index_tuples = list(zip(selected_2d_indices[0], selected_2d_indices[1]))

    return selected_index_tuples

def corrupt_non_normal_probabilities(prob_row,perc):
    corrupted_indexes = select_random_index_tuples(prob_row, percentage = perc)
    for corrupted_index in corrupted_indexes:
        max = np.max(prob_row[corrupted_index[0]][corrupted_index[1]])
        argmax = np.argmax(prob_row[corrupted_index[0]][corrupted_index[1]])
        new_value1,new_value2 = distribute_value(max)
        prob_row[corrupted_index[0]][corrupted_index[1]][0] = new_value1 + prob_row[corrupted_index[0]][corrupted_index[1]][0]
        prob_row[corrupted_index[0]][corrupted_index[1]][argmax] = new_value2
    return prob_row

import matplotlib.pyplot as plt

def visualize_exact_correct_predictions(data, y):
    """
    Visualizes statistics of samples with exactly N correct argmax predictions across classifiers.
    
    Parameters:
    data (numpy.ndarray): 3D array where the 1st axis is samples, 
                          the 2nd axis is classifiers, 
                          and the 3rd axis is probabilities of each class.
    y (numpy.ndarray): 1D array of true class labels for each sample.
    """
    # Number of samples, classifiers, and classes
    num_samples, num_classifiers, num_classes = data.shape
    
    # Initialize an array to store the number of correct predictions per sample
    correct_predictions_per_sample = np.zeros(num_samples)
    
    # Iterate over samples
    for i in range(num_samples):
        # Count the number of classifiers that have the correct argmax prediction for this sample
        if y[i] != 0:
            correct_predictions = np.sum(np.argmax(data[i], axis=1) != 0)
        else:
            correct_predictions = np.sum(np.argmax(data[i], axis=1) == 0)
        correct_predictions_per_sample[i] = correct_predictions
    
    # Prepare data for plotting: counting samples with exactly n correct predictions
    exact_correct_prediction_counts = np.zeros(num_classifiers + 1)
    
    for count in range(0, num_classifiers + 1):  # Start from 0 to include 0 correct predictions
        exact_correct_prediction_counts[count] = np.sum(correct_predictions_per_sample == count)

    # Convert counts to percentages
    exact_correct_prediction_counts = (100 * exact_correct_prediction_counts / data.shape[0])
    
    # Plot the statistics
    plt.figure(figsize=(6, 3))
    plt.bar(range(0, num_classifiers + 1), exact_correct_prediction_counts, color='skyblue')  # Include 0 in range
    plt.xlabel('Number of Exact Correct Argmax Predictions')
    plt.ylabel('Percentage of Samples (%)')  # Updated to show percentages
    plt.title('Percentage of Samples with Exactly N Correct Argmax Predictions')
    plt.xticks(range(0, num_classifiers + 1))  # Include 0 in xticks
    plt.show()
    
def distribute_number(number, x):
    """
    Distributes a number into x smaller values such that their sum equals the original number.

    Parameters:
    - number: The number to be distributed.
    - x: The number of smaller values.

    Returns:
    - A list of x values whose sum equals the original number.
    """
    # Generate x-1 random breakpoints between 0 and the number
    breakpoints = np.sort(np.random.uniform(0, number, x-1))

    # Add the start and end points
    breakpoints = np.concatenate(([0], breakpoints, [number]))

    # The differences between consecutive breakpoints give the distributed values
    distributed_values = np.diff(breakpoints)

    return distributed_values


def select_exponential_values(x, classes, scale=4):
    """
    Selects x values between 1 and classes based on an exponential distribution.

    Parameters:
    - x: The number of values to select.
    - classes: The upper limit for the values (inclusive).
    - scale: The scale parameter (1/lambda) of the exponential distribution.

    Returns:
    - A list of x values between 1 and classes.
    """

    # Sample from the Poisson distribution
    exponential_values = np.random.poisson(scale, x)
    # print(exponential_values)
    # Normalize the values to the range [0, classes-1] and shift to [1, classes]
    
    selected_values = (np.clip(exponential_values, 1, classes))
    return selected_values

def corrupt_probabilities(prob_row,Classes,perc):
    
    corrupted_indexes = select_random_index_tuples(prob_row,percentage = perc)
    for corrupted_index in corrupted_indexes:
        prob_of_zero = prob_row[corrupted_index[0]][corrupted_index[1]][0]
        # spikes = np.random.randint(1, 2)
        spikes = 2
        distribute_zero = distribute_number(prob_of_zero,spikes)
        classes_modified = select_exponential_values(spikes-1,Classes-1)
        prob_row[corrupted_index[0]][corrupted_index[1]][0] = distribute_zero[0]
        for i in range(len(classes_modified)):
            prob_row[corrupted_index[0]][corrupted_index[1]][classes_modified[i]] = prob_row[corrupted_index[0]][corrupted_index[1]][classes_modified[i]] + distribute_zero[i+1]

    return prob_row

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
            raise ValueError(f"The new_class_count ({new_class_count}) should be greater than the original class count ({original_class_count}).")
        
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

class ModelTrainer:
    def __init__(self, folder_path, num_models, num_epochs, batch_size, final_dataset, input_dim, Classes, NOS, NO_WC, device,model_name = "DTF"):
        self.folder_path = folder_path
        self.num_models = num_models
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.final_dataset = final_dataset
        self.input_dim = input_dim
        self.Classes = Classes
        self.NOS = NOS
        self.NO_WC = NO_WC
        self.device = device
        self.model_name = model_name
        
        os.makedirs(self.folder_path, exist_ok=True)

    def train_and_save_models(self):
        with pd.ExcelWriter(os.path.join(self.folder_path, 'metrics.xlsx')) as writer:
            for model_num in range(1, self.num_models + 1):
                model_folder = os.path.join(self.folder_path, f"model_{model_num}")
                os.makedirs(model_folder, exist_ok=True)
                
                train_dataset, test_dataset = self.split_dataset(self.final_dataset)
                train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
                test_dataloader = DataLoader(test_dataset, batch_size=1500, shuffle=True, num_workers=8, pin_memory=True)
                if self.model_name == "DTF":
                    model = DTF(self.input_dim, self.Classes, self.NOS, self.NO_WC,self.device).to(self.device)
                elif self.model_name == "TF":
                    model = TF(self.input_dim, self.Classes).to(self.device)
                    
                    
                loss_fn = nn.CrossEntropyLoss()
                optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)
                lr_scheduler = self.create_lr_scheduler(optimizer, self.num_epochs, len(train_dataloader))
                
                train_losses, metrics, loss_log = self.train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, lr_scheduler, model_num)
                self.save_metrics(metrics, writer, model_num, loss_log)
                self.save_model(model, model_num)
                self.plot_metrics(metrics, model_folder, loss_log)

    def split_dataset(self, dataset, test_size=0.2):
        dataset_size = len(dataset)
        test_size = int(test_size * dataset_size)
        train_size = dataset_size - test_size
        return random_split(dataset, [train_size, test_size])

    def create_lr_scheduler(self, optimizer, num_epochs, num_batches):
        num_training_steps = num_epochs * num_batches
        num_warmup_steps = int(0.1 * num_training_steps)
        return get_scheduler("cosine", optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    def train_model(self, model, train_dataloader, test_dataloader, loss_fn, optimizer, lr_scheduler, model_num):
        train_losses = []
        metrics = {"Epoch": [], "Avg Loss": [], "Accuracy": [], "Precision": [], "Recall": [], "Test Loss": []}
        loss_log = {"Iteration": [], "Loss": []}

        progress_bar = tqdm(total=len(train_dataloader) * self.num_epochs, desc=f"Model {model_num}, Epoch 0")
        
        for epoch in range(self.num_epochs):
            total_loss = self.train_one_epoch(model, train_dataloader, loss_fn, optimizer, lr_scheduler, loss_log, progress_bar, epoch, model_num)
            avg_loss = total_loss / len(train_dataloader)
            train_losses.append(avg_loss)
            
            test_loss, accuracy, precision, recall = self.evaluate_model(model, test_dataloader, loss_fn)
            print(f"accuracy: {accuracy}_precision: {precision}_test_loss: {test_loss}_recall: {recall}")
            self.log_metrics(metrics, epoch, avg_loss, accuracy, precision, recall, test_loss)
        
        progress_bar.close()
        return train_losses, metrics, loss_log

    def train_one_epoch(self, model, dataloader, loss_fn, optimizer, lr_scheduler, loss_log, progress_bar, epoch, model_num):
        model.train()
        total_loss = 0
        
        if self.model_name == "TF":
            for batch_idx, (data1, target) in enumerate(dataloader):
                data1, target = data1.to(self.device), target.squeeze().long().to(self.device)
                
                output = model(data1)
                loss = 0
                
                # output1 = output[:,1::2]
                # output2 = output[:,0::2]
                
                for i in range(target.shape[1]):
                    loss += loss_fn(output[:,i], target[:,i])
                    # loss += loss_fn(output2[:,i], target[:,i])
                loss = loss / (target.shape[1])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_description(f"Model {model_num}, Epoch {epoch}, Loss: {loss.item():.4f}")
                progress_bar.update(1)
                
                loss_log["Iteration"].append(batch_idx + epoch * len(dataloader))
                loss_log["Loss"].append(loss.item())
            
            return total_loss
        
        if self.model_name == "DTF":
            for batch_idx, (data1, data2, target) in enumerate(dataloader):
                data1, data2, target = data1.to(self.device), data2.to(self.device), target.squeeze().long().to(self.device)
                
                output = model(data1, data2)
                loss = 0
                
                # output1 = output[:,1::2]
                # output2 = output[:,0::2]
                
                for i in range(target.shape[1]):
                    loss += loss_fn(output[:,i], target[:,i])
                    # loss += loss_fn(output2[:,i], target[:,i])
                loss = loss / (target.shape[1])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_description(f"Model {model_num}, Epoch {epoch}, Loss: {loss.item():.4f}")
                progress_bar.update(1)
                
                loss_log["Iteration"].append(batch_idx + epoch * len(dataloader))
                loss_log["Loss"].append(loss.item())
            
            return total_loss

    def evaluate_model(self, model, dataloader, loss_fn):
        model.eval()
        total_loss = 0
        true_labels, predicted_labels = [], []

        with torch.no_grad():
            if self.model_name == "DTF":
                for data1, data2, target in dataloader:
                    data1, data2, target = data1.to(self.device), data2.to(self.device), target.squeeze().long().to(self.device)
                    output = model(data1, data2)
                    loss = 0
                
                    # output1 = output[:,1::2]
                    # output2 = output[:,0::2]
                    
                    for i in range(target.shape[1]):
                        loss += loss_fn(output[:,i], target[:,i])
                        # loss += loss_fn(output2[:,i], target[:,i])
                    loss = loss / (2*target.shape[1])
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(output, 2)
                    true_labels.extend(target.cpu().numpy())
                    predicted_labels.extend(predicted.cpu().numpy())
            elif self.model_name == "TF":
                for data1, target in dataloader:
                    data1, target = data1.to(self.device), target.squeeze().long().to(self.device)
                    output = model(data1)
                    loss = 0
                
                    # output1 = output[:,1::2]
                    # output2 = output[:,0::2]
                    
                    for i in range(target.shape[1]):
                        loss += loss_fn(output[:,i], target[:,i])
                        # loss += loss_fn(output2[:,i], target[:,i])
                    loss = loss / (2*target.shape[1])
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(output, 2)
                    true_labels.extend(target.cpu().numpy())
                    predicted_labels.extend(predicted.cpu().numpy())

        test_loss = total_loss / len(dataloader)
        accuracy, precision, recall = self.calculate_metrics(np.array(true_labels), np.array(predicted_labels))
        return test_loss, accuracy, precision, recall

    def calculate_metrics(self, true_labels, predicted_labels):
        accuracies, precisions, recalls = [], [], []

        for i in range(true_labels.shape[1]):
            accuracies.append(accuracy_score(true_labels[:, i], predicted_labels[:, i]))
            precisions.append(precision_score(true_labels[:, i], predicted_labels[:, i], average='macro', zero_division=0))
            recalls.append(recall_score(true_labels[:, i], predicted_labels[:, i], average='macro', zero_division=0))

        avg_accuracy = sum(accuracies) / true_labels.shape[1]
        avg_precision = sum(precisions) / true_labels.shape[1]
        avg_recall = sum(recalls) / true_labels.shape[1]
        return avg_accuracy, avg_precision, avg_recall

    def log_metrics(self, metrics, epoch, avg_loss, accuracy, precision, recall, test_loss):
        metrics["Epoch"].append(epoch)
        metrics["Avg Loss"].append(avg_loss)
        metrics["Accuracy"].append(accuracy)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["Test Loss"].append(test_loss)

    def save_metrics(self, metrics, writer, model_num, loss_log):
        pd.DataFrame(metrics).to_excel(writer, sheet_name=f"Model_{model_num}_metrics", index=False)
        pd.DataFrame(loss_log).to_excel(writer, sheet_name=f"Model_{model_num}_loss", index=False)

    def save_model(self, model, model_num):
        model_file = os.path.join(self.folder_path, f"Model_{model_num}.pth")
        torch.save(model.state_dict(), model_file)

    def plot_metrics(self, metrics, model_folder, loss_log):
        self.plot_and_save(metrics["Epoch"], metrics["Test Loss"], "Epoch", "Test Loss", os.path.join(model_folder, 'test_loss_plot.png'))
        self.plot_and_save(metrics["Epoch"], metrics["Avg Loss"], "Epoch", "Avg Loss", os.path.join(model_folder, 'avg_loss_plot.png'))
        self.plot_and_save(metrics["Epoch"], metrics["Accuracy"], "Epoch", "Accuracy", os.path.join(model_folder, 'accuracy_plot.png'))
        self.plot_and_save(metrics["Epoch"], metrics["Precision"], "Epoch", "Precision", os.path.join(model_folder, 'precision_plot.png'))
        self.plot_and_save(metrics["Epoch"], metrics["Recall"], "Epoch", "Recall", os.path.join(model_folder, 'recall_plot.png'))
        self.plot_and_save(loss_log["Iteration"], loss_log["Loss"], "Iteration", "Loss", os.path.join(model_folder, 'loss_plot.png'))

    def plot_and_save(self, x, y, xlabel, ylabel, filepath):
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label=ylabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(filepath)
        plt.close()
        

