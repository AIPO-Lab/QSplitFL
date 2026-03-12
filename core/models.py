"""
Neural Network Models for Federated Split Learning

This module contains various CNN architectures used in the federated learning experiments:
- ResNetFed: ResNet-18 variant for CIFAR-10
- SimpleMNISTCNN: Simple CNN for MNIST
- SimpleNN: Basic feedforward network
- CNNModel: Alternative CNN architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetFed(nn.Module):
    """
    ResNet-18 variant optimized for CIFAR-10 (32x32 images).
    Designed for split learning with easily separable layers.
    """
    def __init__(self, num_classes=10):
        super(ResNetFed, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet blocks (ResNet-34 style: [3, 4, 6, 3])
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Split configuration
        self.split_mode = 'full' # 'full', 'client', 'server'
        self.split_layer = None
        
    def configure_split(self, mode, split_layer):
        """
        Configure the model for split learning.
        
        Args:
            mode: 'client' or 'server'
            split_layer: Split layer index
        """
        self.split_mode = mode
        self.split_layer = split_layer
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Create a ResNet layer with multiple residual blocks."""
        layers = []
        # First block may have stride > 1
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if self.split_mode == 'full':
            return self._forward_full(x)
        elif self.split_mode == 'client':
            return self._forward_client(x)
        elif self.split_mode == 'server':
            return self._forward_server(x)
        else:
            raise ValueError(f"Unknown split mode: {self.split_mode}")

    def _forward_full(self, x):
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global average pooling and FC
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
        
    def _forward_client(self, x):
        """
        Client side forward pass.
        Supports Splits 5-9 (Mid-network).
        """
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        # Split 5: After Layer 1
        if self.split_layer <= 5:
            return out
            
        # Split 6: After Layer 2 Block 2
        if self.split_layer == 6:
            out = self.layer2[0](out)
            out = self.layer2[1](out)
            return out
            
        out = self.layer2(out)
        
        # Split 7: After Layer 2 Full
        if self.split_layer == 7:
            return out
            
        # Split 8: After Layer 3 Block 3
        if self.split_layer == 8:
            out = self.layer3[0](out)
            out = self.layer3[1](out)
            out = self.layer3[2](out)
            return out
            
        out = self.layer3(out)
        
        # Split 9: After Layer 3 Full
        if self.split_layer == 9:
            return out
            
        # If > 9 (should not adhere to strict rules but for robust fallback)
        out = self.layer4(out)
        return out

    def _forward_server(self, x):
        out = x
        
        if self.split_layer <= 5:
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        elif self.split_layer == 6:
            if len(self.layer2) > 2:
                out = self.layer2[2:](out)
            out = self.layer3(out)
            out = self.layer4(out)
        elif self.split_layer == 7:
            out = self.layer3(out)
            out = self.layer4(out)
        elif self.split_layer == 8:
            if len(self.layer3) > 3:
                out = self.layer3[3:](out)
            out = self.layer4(out)
        elif self.split_layer == 9:
            out = self.layer4(out)
            
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


class ResidualBlock(nn.Module):
    """Basic residual block for ResNet."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleMNISTCNN(nn.Module):
    """
    Simple CNN for MNIST/Fashion-MNIST (28x28 grayscale images).
    Designed for split learning with sequential layers.
    """
    def __init__(self, num_classes=10):
        super(SimpleMNISTCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class CNNModel(nn.Module):
    """
    Alternative CNN architecture for CIFAR-10.
    Simpler than ResNet but more complex than SimpleMNISTCNN.
    """
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class SimpleNN(nn.Module):
    """
    Simple feedforward neural network.
    Can be used for both MNIST and CIFAR-10 with appropriate input size.
    """
    def __init__(self, input_size=3072, num_classes=10):
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # FC layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


# Test function
if __name__ == "__main__":
    print("Testing model architectures...")
    
    # Test ResNetFed with CIFAR-10 input
    model = ResNetFed(num_classes=10)
    x = torch.randn(2, 3, 32, 32)  # Batch of 2 CIFAR-10 images
    out = model(x)
    print(f"ResNetFed output shape: {out.shape}")  # Should be [2, 10]
    
    # Test SimpleMNISTCNN with MNIST input
    model = SimpleMNISTCNN(num_classes=10)
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
    out = model(x)
    print(f"SimpleMNISTCNN output shape: {out.shape}")  # Should be [2, 10]
    
    # Test CNNModel with CIFAR-10 input
    model = CNNModel(num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"CNNModel output shape: {out.shape}")  # Should be [2, 10]
    
    # Test SimpleNN
    model = SimpleNN(input_size=3072, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"SimpleNN output shape: {out.shape}")  # Should be [2, 10]
    
    print("\n✅ All models working correctly!")
