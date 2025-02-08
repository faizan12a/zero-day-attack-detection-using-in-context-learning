import torch
import torch.nn as nn


# Define a 1D Residual Block
class ResidualBlock1D(nn.Module):
    def __init__(self, filters, kernel_size=3, strides=1, dropout_rate=0.4):
        super(ResidualBlock1D, self).__init__()
        
        # First 1x1 Convolution (Bottleneck)
        self.conv1 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=1, stride=strides, padding=0)
        self.bn1 = nn.BatchNorm1d(filters)
        
        # Second 3x3 Convolution
        self.conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(filters)
        
        # Third 1x1 Convolution (Restore Dimensionality)
        self.conv3 = nn.Conv1d(in_channels=filters, out_channels=4 * filters, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(4 * filters)
        
        # Shortcut (Identity or Convolution)
        self.shortcut_conv = nn.Conv1d(in_channels=filters, out_channels=4 * filters, kernel_size=1, stride=strides, padding=0)
        self.shortcut_bn = nn.BatchNorm1d(4 * filters)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        # Main path (Residual Block)
        residual = x
        
        # First Convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Second Convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        # Third Convolution
        x = self.conv3(x)
        x = self.bn3(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Shortcut path
        shortcut = self.shortcut_conv(residual)
        shortcut = self.shortcut_bn(shortcut)
        
        # Add shortcut to main path
        x += shortcut
        x = self.activation(x)
        
        return x


class MyNN(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.1):
        super(MyNN, self).__init__()
        # Initial convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(58)
        self.activation = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual Blocks (you can add more stages like in ResNet)
        self.res_block1 = ResidualBlock1D(filters=32, kernel_size=3, strides=1, dropout_rate=0.3)
        self.res_block2 = ResidualBlock1D(filters=128, kernel_size=3, strides=1, dropout_rate=0.2)        
        self.res_block3 = ResidualBlock1D(filters=512, kernel_size=3, strides=1, dropout_rate=0.2)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(2048, n_classes)
        
        # Dropout layer before the final output
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # Initial convolution + activation + pooling
        x = self.bn1(x)
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.dropout(x)
        x = self.res_block3(x)
        x = self.dropout(x)
        
        features = self.global_avg_pool(x)  # These are the extracted features
        features = torch.flatten(features, 1)  # Flatten the output for the fully connected layer
    
        # Fully connected layer
        x = self.dropout(features)
        x = self.fc(x)
        return x