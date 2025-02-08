import numpy as np
import pandas as pd
import os
import json
from itertools import combinations_with_replacement
from sklearn.preprocessing import StandardScaler
import argparse

def generate_combinations_with_repetitions(paths, total_combinations_needed, normal_percentage=0.99999):
    all_combinations = []
    labels = []
    i = 1
    j = 1
    total_combinations = 0

    while total_combinations < total_combinations_needed:
        for combo in combinations_with_replacement(paths, i):
            if (len(set(combo)) == 1 and len(combo) > 1) or (len(combo) > 1 and 0 in combo):
                continue

            combo_count = combo.count(0)
            if combo_count / len(combo) > normal_percentage:
                label = 0
            else:
                label = j
                j += 1

            all_combinations.append(combo)
            labels.append(label)
            total_combinations += 1

            if total_combinations == total_combinations_needed:
                return all_combinations, labels
        i += 1

def distribute_length_zipfian(total_length, num_datasets, ZConstant):
    ranks = np.arange(1, num_datasets + 1)
    zipfian_distribution = 1 / np.power(ranks, ZConstant)
    zipfian_distribution /= zipfian_distribution.sum()

    raw_lengths = zipfian_distribution * total_length
    lengths = np.floor(raw_lengths).astype(int)
    difference = total_length - lengths.sum()
    if difference > 0:
        indices = np.random.choice(num_datasets, difference, replace=False)
        lengths[indices] += 1

    return lengths

def Dataset_Load_Multi_Zipfian(Combinations, Label, Data, Length, ZConstant, mode=0):
    if mode == 1:
        for i in range(len(Data)):
            scalar = StandardScaler()
            temp_data = np.copy(Data[i])
            temp_data[:,0:46] = scalar.fit_transform(temp_data[:,0:46])
            Data[i] = temp_data
    Dataset_Lengths = distribute_length_zipfian(Length, len(Label), ZConstant)
    Total_Dataset_Length = np.sum(Dataset_Lengths)
    X = np.zeros((Total_Dataset_Length, Data[0].shape[1]))
    y = np.zeros((Total_Dataset_Length,))
    index = 0

    for i, combination in enumerate(Combinations):
        Length_Combo = Dataset_Lengths[i]
        X_Combo = np.zeros((Length_Combo, Data[0].shape[1]))

        for dataset in combination:
            Packet_Data = Data[dataset]
            Data1 = Packet_Data[np.random.choice(Packet_Data.shape[0], size=Length_Combo, replace=False)]
            X_Combo += Data1

        y_Combo = np.tile(Label[i], Length_Combo)
        X_Combo = X_Combo / len(combination)

        X[index:index+Length_Combo] = X_Combo
        y[index:index+Length_Combo] = y_Combo
        index += Length_Combo

    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]

    return X, y

if __name__ == "__main__":
    # Load configuration from JSON
    parser = argparse.ArgumentParser(description="Load CSV files dynamically using a JSON config")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument('--tasks',type=int,required=True)
    parser.add_argument('--Z_Constant',type=float,required=True)
    parser.add_argument('--Length',type=int,required=True)
    parser.add_argument('--base_path',type=str,required=True)
    
    args = parser.parse_args()

    # Read the JSON configuration file
    with open(args.config, 'r') as file:
        config = json.load(file)


    base_path = args.base_path
    data_path = config["data_path"]
    save_path = config["save_path"]
    tasks = args.tasks
    Z_Constant = args.Z_Constant
    Length = args.Length
    mode = config["mode"]

    # Load datasets
    data1 = np.load(base_path + data_path + 'GOOSE_normal_train.npy')
    data2 = np.load(base_path + data_path + 'GOOSE_highStNum_train.npy')
    data3 = np.load(base_path + data_path + 'GOOSE_inversereplay_train.npy')
    data4 = np.load(base_path + data_path + 'GOOSE_injection_train.npy')
    data5 = np.load(base_path + data_path + 'SV_highStNum_train.npy')
    data6 = np.load(base_path + data_path + 'SV_injection_train.npy')
    paths = [0, 1, 2, 3, 4, 5]  # 0 is normal, others represent attacks
    Data = [data1, data2, data3, data4, data5, data6]

    # Generate combinations and labels
    combinations, labels = generate_combinations_with_repetitions(paths, tasks)
    X, y = Dataset_Load_Multi_Zipfian(combinations, labels, Data, Length, Z_Constant,mode=mode)

    # Save results
    tasks_folder = os.path.join(base_path + save_path, f"tasks_{tasks}")
    os.makedirs(tasks_folder, exist_ok=True)

    folder_name = f"zipfian_constant_{Z_Constant}"
    save_folder = os.path.join(tasks_folder, folder_name)


    os.makedirs(save_folder, exist_ok=True)

    np.save(os.path.join(save_folder, 'X.npy'), X)
    np.save(os.path.join(save_folder, 'y.npy'), y)
    print(f"Arrays saved in {save_folder}")
