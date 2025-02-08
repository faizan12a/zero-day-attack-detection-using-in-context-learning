import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import argparse
import json

def load_dataframes(base_path, data_path, dataset_type='train'):
    """
    Load dataframes from the specified path and dataset type.

    Parameters:
        base_path: The base directory path.
        data_path: Relative path to the dataset directory.
        dataset_type: Type of the dataset ('train' or 'test').

    Returns:
        dataframes: A list of loaded Pandas DataFrames.
    """
    # List of base file names for the datasets
    file_names = [
        f'GOOSE_normal_{dataset_type}.csv',
        f'GOOSE_highStNum_{dataset_type}.csv',
        f'GOOSE_inversereplay_{dataset_type}.csv',
        f'GOOSE_injection_{dataset_type}.csv',
        f'SV_highStNum_{dataset_type}.csv',
        f'SV_injection_{dataset_type}.csv',
        f'GOOSE_masqueradefakenormal_{dataset_type}.csv',
        f'GOOSE_masqueradefakefault_{dataset_type}.csv',
        f'GOOSE_randomreplay_{dataset_type}.csv',
        f'GOOSE_poisonedhighrate_{dataset_type}.csv',
    ]

    # Construct the full paths and load data
    dataframes = []
    for file_name in file_names:
        print(f"Loading file: {file_name}")
        full_path = os.path.join(base_path, data_path, file_name)
        try:
            df = pd.read_csv(full_path)
            df.columns = range(df.shape[1])  # Standardize column names
            dataframes.append(df)
        except FileNotFoundError:
            print(f"File not found: {full_path}. Skipping.")

    return dataframes

def resample_dataframe(df, length=15000):
    """
    Resample the DataFrame to generate new samples if the desired length is greater
    than the number of samples in the DataFrame, and append them to the original DataFrame.
    """
    if len(df) >= length:
        return df

    num_features = len(df.columns)
    resampled_data = []

    for _ in range(length - len(df)):
        new_sample = []
        for column in df.columns:
            if df[column].dtype == 'object':
                most_common_value = df[column].mode().iloc[0]
                new_sample.append(most_common_value)
            else:
                new_sample.append(np.random.uniform(df[column].min(), df[column].max()))
        resampled_data.append(new_sample)

    resampled_df = pd.DataFrame(resampled_data, columns=df.columns)
    return pd.concat([df, resampled_df], ignore_index=True)

# Function to replace missing values
def replace_missing_values(df):
    for dtype in ['object', 'number']:
        columns = df.select_dtypes(include=[dtype]).columns
        for column in columns:
            replacement = df[column].mode()[0] if dtype == 'object' else df[column].mean()
            df[column].fillna(replacement, inplace=True)
    return df

# Data preprocessing function
def df_preprocessing(df):
    string_df = replace_missing_values(df.select_dtypes(include=['object']))
    numeric_df = replace_missing_values(df.select_dtypes(include=[int, float]))
    numeric_df = numeric_df.dropna()
    return pd.concat([numeric_df, string_df], axis=1)

# Function to replace unknown categories
def replace_with_unknown(test_data, known_categories):
    for col in test_data.columns:
        test_data[col] = test_data[col].apply(lambda x: x if x in known_categories[col] else 'unknown')
    return test_data

# Function to encode and concatenate DataFrame
def encode_and_concatenate(df, encoder, known_categories):
    string_columns_test = replace_with_unknown(df.select_dtypes(include=['object']), known_categories)
    encoded_test = encoder.transform(string_columns_test)
    encoded_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(string_columns_test.columns.astype(str)))
    numeric_columns = df.select_dtypes(exclude=['object'])
    return pd.concat([numeric_columns.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

def process_data(data_path, dataset_length_min, base_path, save_path, file_suffix,encoder = None):
    """
    Processes data by loading, resampling, preprocessing, encoding, and saving.
    
    Parameters:
        data_path: Path to the data files.
        dataset_length_min: Minimum length for resampling.
        base_path: Base directory for saving files.
        save_path: Directory to save processed files.
        file_suffix: Suffix for the saved file names ('train' or 'test').
    """
    if file_suffix == 'train':
        save_path = save_path + 'training_preprocessed/'
    else:
        save_path = save_path + 'testing_preprocessed/'
        

    
    # Load dataframes
    dataframes = load_dataframes(base_path, data_path,file_suffix)

    # Drop constant columns
    numeric_columns = dataframes[0].select_dtypes(include=[int, float]).columns
    data_frame_normal = dataframes[0][numeric_columns]
    columns_to_drop = (data_frame_normal.max() - data_frame_normal.min() == 0)
    columns_to_drop = columns_to_drop[columns_to_drop].index
    columns_to_drop = list(columns_to_drop)

    # Add the first column by index
    columns_to_drop.append(dataframes[0].columns[0])

    # Apply column dropping to all dataframes
    dataframes = [df.drop(columns=columns_to_drop) for df in dataframes]

    # Resample the dataframes
    if file_suffix == 'train':
        resampled_dfs = [resample_dataframe(df, dataset_length_min) for df in dataframes]
    else:
        resampled_dfs = dataframes

    # Preprocess the resampled dataframes
    preprocessed_dfs = [df_preprocessing(df) for df in resampled_dfs]

    # One-hot encoding
    string_columns = preprocessed_dfs[0].select_dtypes(include=['object'])
    string_columns = string_columns.apply(lambda col: col.replace('nan', 'unknown'))
    if encoder == None:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(string_columns)
    known_categories = {col: string_columns[col].unique() for col in string_columns.columns}

        # Encode and concatenate
    encoded_dfs = [encode_and_concatenate(df, encoder, known_categories) for df in preprocessed_dfs]

    # Save the processed DataFrames
    save_directory = os.path.join(base_path, save_path)
    os.makedirs(save_directory, exist_ok=True)
    file_names = [
        f'GOOSE_normal_{file_suffix}.npy',
        f'GOOSE_highStNum_{file_suffix}.npy',
        f'GOOSE_inversereplay_{file_suffix}.npy',
        f'GOOSE_injection_{file_suffix}.npy',
        f'SV_highStNum_{file_suffix}.npy',
        f'SV_injection_{file_suffix}.npy',
        f'GOOSE_masqueradefakenormal_{file_suffix}.npy',
        f'GOOSE_masqueradefakefault_{file_suffix}.npy',
        f'GOOSE_randomreplay_{file_suffix}.npy',
        f'GOOSE_poisonedhighrate_{file_suffix}.npy',
    ]

    for file_name, encoded_df in zip(file_names, encoded_dfs):
        path = os.path.join(save_directory, file_name)
        np.save(path, encoded_df.values)

    print(f"All {file_suffix} files saved successfully!")
    return encoder
    
if __name__ == "__main__":
    # Setup argparse
    parser = argparse.ArgumentParser(description="Load CSV files dynamically using a JSON config")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument('--base_path', type=str, required=True)

    args = parser.parse_args()

    # Read the JSON configuration file
    with open(args.config, 'r') as file:
        config = json.load(file)

    # Extract configuration values
    base_path = args.base_path
    data_path_train = config['data_path_train']
    data_path_test = config['data_path_test']
    dataset_length_min = config['dataset_length_min']
    save_path = config['save_path']

    # Example usage of the parameters
    print(f"Base Path: {base_path}")
    print(f"Data Path Training: {data_path_train}")
    print(f"Data Path Testing: {data_path_test}")
    print(f"Minimum Dataset Length: {dataset_length_min}")
    print(f"Save Path: {save_path}")
    
    # Process training data
    train_encoder = process_data(data_path_train, dataset_length_min, base_path, save_path, 'train')

    # Process testing data
    process_data(data_path_test, dataset_length_min, base_path, save_path, 'test',encoder=train_encoder)

    print("All files saved successfully!")
