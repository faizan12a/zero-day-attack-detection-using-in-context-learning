import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNModel(nn.Module):
    def __init__(self, input_shape):
        super(DNNModel, self).__init__()
        self.input_dim = input_shape[0]  # Extracting the number of input features

        # Layer 0-1: Dense -> Batch Normalization -> Dropout
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.5)

        # Layer 1-2: Dense -> Batch Normalization -> Dropout
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)

        # Layer 2-3: Dense -> Batch Normalization -> Dropout
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)

        # Layer 4-5: Dense -> Batch Normalization -> Dropout
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.5)

        # Final Dense layer for binary classification
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        # Layer 0-1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Layer 1-2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Layer 2-3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Layer 4-5
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        # Output layer
        x = self.output(x)
        x = torch.sigmoid(x)  # For binary classification

        return x

class CNNModel(nn.Module):
    def __init__(self, input_shape):
        super(CNNModel, self).__init__()
        # Assuming input_shape is (features, time_steps)
        self.features = input_shape[0]
        self.time_steps = input_shape[1]

        # First Conv1D layer
        self.conv1 = nn.Conv1d(in_channels=self.features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)  # BatchNorm for 32 output channels
        self.pool1 = nn.MaxPool1d(kernel_size=2, padding=1)

        # Second Conv1D layer
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)  # BatchNorm for 64 output channels
        self.pool2 = nn.MaxPool1d(kernel_size=2, padding=1)

        # Third Conv1D layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)  # BatchNorm for 128 output channels

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        # Adjust input size for the first FC layer based on flattened output size
        self.fc1 = nn.Linear(128 * ((self.time_steps + 3) // 4), 128)
        self.bn_fc1 = nn.BatchNorm1d(128)  # BatchNorm after first FC layer
        self.fc2 = nn.Linear(128, 1)  # Final output layer

    def forward(self, x):
        # Input shape: (batch_size, features, time_steps)
        # PyTorch Conv1D expects (batch_size, features, time_steps), so ensure correct dimensions
        x = x.unsqueeze(2)
        # x = x.permute(0, 2, 1)  # (batch_size, features, time_steps) -> (batch_size, time_steps, features)

        # First Conv1D -> BatchNorm -> ReLU -> MaxPool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Second Conv1D -> BatchNorm -> ReLU -> MaxPool
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Third Conv1D -> BatchNorm -> ReLU
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers with BatchNorm and Dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Binary classification

        return x


class RNNModel(nn.Module):
    def __init__(self, input_shape):
        super(RNNModel, self).__init__()
        self.input_dim = input_shape[0]  # Number of features (input_shape = (time_steps, features))
        self.time_steps = input_shape[1]

        # RNN layers
        self.rnn1 = nn.RNN(input_size=self.input_dim, hidden_size=64, batch_first=True)
        self.bn1 = nn.BatchNorm1d(self.time_steps)

        self.rnn2 = nn.RNN(input_size=64, hidden_size=64, batch_first=True)
        self.bn2 = nn.BatchNorm1d(self.time_steps)
        self.dropout2 = nn.Dropout(0.5)

        self.rnn3 = nn.RNN(input_size=64, hidden_size=64, batch_first=True)
        self.bn3 = nn.BatchNorm1d(self.time_steps)
        self.dropout3 = nn.Dropout(0.5)

        self.rnn4 = nn.RNN(input_size=64, hidden_size=32, batch_first=True)
        self.bn4 = nn.BatchNorm1d(self.time_steps)
        self.dropout4 = nn.Dropout(0.5)

        self.rnn5 = nn.RNN(input_size=32, hidden_size=32, batch_first=True)
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout5 = nn.Dropout(0.5)

        # Dense layer
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # Pass through the first RNN layer
        x = x.unsqueeze(2)
        x = x.permute(0,2,1)
        x, _ = self.rnn1(x)
        x = self.bn1(x)

        # Pass through the second RNN layer
        x, _ = self.rnn2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        # Pass through the third RNN layer
        x, _ = self.rnn3(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        # Pass through the fourth RNN layer
        x, _ = self.rnn4(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        # Pass through the fifth RNN layer
        x, _ = self.rnn5(x)
        x = x.permute(0,2,1)
        x = self.bn5(x)
        x = x[:, :, -1]  # Keep only the last output (return_sequences=False)
        
        x = self.dropout5(x)

        # Dense layer
        x = torch.sigmoid(self.fc(x))  # Binary classification

        return x

class LSTMModel(nn.Module):
    def __init__(self, input_shape):
        super(LSTMModel, self).__init__()
        self.input_dim = input_shape[0]  # Number of features (input_shape = (time_steps, features))
        self.time_steps = input_shape[1]

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=self.input_dim, hidden_size=128, batch_first=True)
        self.bn1 = nn.BatchNorm1d(self.time_steps)
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.bn2 = nn.BatchNorm1d(self.time_steps)
        self.dropout2 = nn.Dropout(0.5)

        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.bn3 = nn.BatchNorm1d(self.time_steps)

        self.lstm4 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.bn4 = nn.BatchNorm1d(self.time_steps)
        self.dropout4 = nn.Dropout(0.5)

        self.lstm5 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout5 = nn.Dropout(0.5)

        # Fully connected output layer
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # First LSTM layer
        x = x.unsqueeze(2)
        x = x.permute(0,2,1)
        x,_ = self.lstm1(x)
        
        x = self.bn1(x)
        x = self.dropout1(x)

        # Second LSTM layer
        x,_ = self.lstm2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        # Third LSTM layer
        x,_ = self.lstm3(x)
        x = self.bn3(x)

        # Fourth LSTM layer
        x,_ = self.lstm4(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        # Fifth LSTM layer
        x,_ = self.lstm5(x)
        x = x.permute(0,2,1)
        
        x = self.bn5(x)
        x = self.dropout5(x)
        x = x[:, :, -1]  # x: (batch_size, 32)

        # Fully connected output layer
        x = torch.sigmoid(self.fc(x))  # Binary classification

        return x
