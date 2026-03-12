"""
Neural Network Models for MNIST Split Learning

Optimized architectures for MNIST dataset (28x28 grayscale images).
All models designed for split learning with easily separable layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMNISTCNN(nn.Module):
    """
    Simple CNN for MNIST - Main model for split learning.
    
    Architecture (10 layers total):
        Layer 0: Conv1 (1→32) + ReLU + Pool
        Layer 1: Conv2 (32→64) + ReLU + Pool  
        Layer 2: Flatten
        Layer 3: FC1 (64*7*7→512) + ReLU
        Layer 4: Dropout(0.25)
        Layer 5: FC2 (512→256) + ReLU
        Layer 6: Dropout(0.25)
        Layer 7: FC3 (256→128) + ReLU
        Layer 8: Dropout(0.25)
        Layer 9: Output (128→10)
    
    Split points: {5, 6, 7, 8, 9} for 10-layer model
    """
    def __init__(self, num_classes=10):
        super(SimpleMNISTCNN, self).__init__()
        
        # Convolutional layers (Layers 0-1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (Layers 3, 5, 7, 9)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Dropout layers (Layers 4, 6, 8)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Layer 0: Conv1 + Pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Layer 1: Conv2 + Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Layer 2: Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Layer 3: FC1
        x = F.relu(self.fc1(x))
        
        # Layer 4: Dropout
        x = self.dropout(x)
        
        # Layer 5: FC2
        x = F.relu(self.fc2(x))
        
        # Layer 6: Dropout
        x = self.dropout(x)
        
        # Layer 7: FC3
        x = F.relu(self.fc3(x))
        
        # Layer 8: Dropout
        x = self.dropout(x)
        
        # Layer 9: Output
        x = self.fc4(x)
        
        return x


# Aliases for compatibility
ResNetFed = SimpleMNISTCNN
CNNModel = SimpleMNISTCNN
SimpleNN = SimpleMNISTCNN


if __name__ == "__main__":
    print("Testing MNIST model architecture...")
    
    model = SimpleMNISTCNN(num_classes=10)
    x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
    out = model(x)
    print(f"✅ SimpleMNISTCNN output shape: {out.shape}")  # Should be [2, 10]
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print("\n✅ Model working correctly!")
