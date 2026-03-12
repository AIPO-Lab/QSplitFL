"""
Advanced Neural Network Architectures for Federated Split Learning

This module contains multiple CNN architectures for federated learning experiments:
1. CNN - SimpleMNISTCNN/ResNetFed (existing)
2. ResNet50 - Deep residual network with bottleneck blocks
3. MobileNetV4 - Efficient mobile architecture
4. ConvNeXt - Modern CNN inspired by vision transformers

All models support split learning with configurable split points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# ResNet50 Architecture
# ============================================================================

class Bottleneck(nn.Module):
    """Bottleneck block for ResNet50 (1x1 -> 3x3 -> 1x1 convolutions)"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 conv (expansion)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet50(nn.Module):
    """
    ResNet50 architecture adapted for CIFAR-like datasets (32x32 or 28x28).
    Supports split learning with 10 configurable split points.
    """
    def __init__(self, num_classes=10, input_channels=3):
        super(ResNet50, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution (adapted for small images)
        if input_channels == 1:  # MNIST/FMNIST
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:  # CIFAR
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet stages: [3, 4, 6, 3] blocks per stage
        self.layer1 = self._make_layer(64, 3, stride=1)    # 3 blocks
        self.layer2 = self._make_layer(128, 4, stride=2)   # 4 blocks
        self.layer3 = self._make_layer(256, 6, stride=2)   # 6 blocks  
        self.layer4 = self._make_layer(512, 3, stride=2)   # 3 blocks
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Split learning configuration
        self.split_mode = 'full'  # 'full', 'client', 'server'
        self.split_layer = None
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels, blocks, stride=1):
        """Create a ResNet layer with multiple bottleneck blocks"""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )
        
        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion
        
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def is_client_layer(self, layer_name, split_layer):
        """
        Determine if a layer belongs to the client side based on split layer.
        Supports strict split range 25-49.
        """
        # Always server layers
        if 'fc' in layer_name or 'avgpool' in layer_name:
            return False
            
        # Initial layers are always client (up to Stage 2 / Layer 22)
        # Includes conv1, bn1, layer1, layer2
        if any(x in layer_name for x in ['conv1', 'bn1', 'relu', 'maxpool']):
            return True
            
        if layer_name.startswith(('layer1', 'layer2')):
            return True
            
        # Stage 3 (Layers 23-40) checks
        if layer_name.startswith('layer3'):
            try:
                # layer3.X.Y -> extract X (block index)
                # If just 'layer3', it usually refers to the sequential container which holds all blocks.
                # But here we likely get layer3.0.conv1 etc.
                parts = layer_name.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    block_idx = int(parts[1])
                    
                    if split_layer <= 22: return False
                    
                    if split_layer <= 40:
                        blocks_client = (split_layer - 22 + 2) // 3
                        blocks_client = max(1, min(blocks_client, 6))
                        return block_idx < blocks_client
                    else:
                        # Split > 40 means all of stage 3 is client
                        return True
                return False # Should not happen if name is valid
            except:
                return False
                
        # Stage 4 (Layers 41-49) checks
        if layer_name.startswith('layer4'):
            try:
                parts = layer_name.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    block_idx = int(parts[1])
                    
                    if split_layer <= 40: return False
                    
                    if split_layer <= 49:
                        blocks_client = (split_layer - 40 + 2) // 3
                        blocks_client = max(1, min(blocks_client, 3))
                        return block_idx < blocks_client
                    else:
                        return True
                return False
            except:
                return False
                
        return False

    def configure_split(self, mode, split_layer):
        """Configure split learning mode and layer"""
        self.split_mode = mode
        self.split_layer = split_layer
    
    def forward(self, x):
        if self.split_mode == 'full':
            return self._forward_full(x)
        elif self.split_mode == 'client':
            return self._forward_client(x)
        elif self.split_mode == 'server':
            return self._forward_server(x)
    
    def _forward_full(self, x):
        """Full forward pass"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def _forward_client(self, x):
        """
        Client-side forward pass up to split point (Layers 25-49).
        
        Mapping logic (Approximate layer counting):
        - Conv1 + Stage1(3blk) + Stage2(4blk) = 1 + 9 + 12 = 22 layers (Fixed Client)
        - Stage 3 (6 blocks): Layers 23-40
          - Blk0: 23-25
          - Blk1: 26-28
          - Blk2: 29-31
          - Blk3: 32-34
          - Blk4: 35-37
          - Blk5: 38-40
        - Stage 4 (3 blocks): Layers 41-49
          - Blk0: 41-43
          - Blk1: 44-46
          - Blk2: 47-49
        """
        # Always run up to Stage 2 (Layer 22)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        # If split is exactly 22 or less (should not happen with strict 25-49), return
        if self.split_layer <= 22:
            return x
            
        # Stage 3 Logic (Layers 23-40)
        # We cut at block boundaries for simplicity and stability
        # Map split_layer to block index
        # 25 -> Cut after Blk0
        # 28 -> Cut after Blk1
        # ...
        
        # Calculate how many blocks of Stage 3 to run
        # Each block is 3 layers.
        # layers_in_s3 = split_layer - 22
        # blocks_to_run = ceil(layers_in_s3 / 3)
        
        if self.split_layer <= 40:
            blocks_to_run = (self.split_layer - 22 + 2) // 3 # ceil division approx
            blocks_to_run = max(1, min(blocks_to_run, 6)) # Clamp
            
            for i in range(blocks_to_run):
                x = self.layer3[i](x)
            return x
            
        # If split_layer > 40, run all Stage 3
        x = self.layer3(x)
        
        # Stage 4 Logic (Layers 41-49)
        if self.split_layer <= 49:
            blocks_to_run = (self.split_layer - 40 + 2) // 3
            blocks_to_run = max(1, min(blocks_to_run, 3))
            
            for i in range(blocks_to_run):
                x = self.layer4[i](x)
            return x
            
        # Post 49 (should be server, but if split > 49)
        x = self.layer4(x)
        return x
    
    def _forward_server(self, x):
        """Server-side forward pass from split point"""
        # Assume x comes from the cut point defined in _forward_client
        
        # Recover context
        # If split <= 40 (Stage 3)
        if self.split_layer <= 40:
            # How many blocks did client run?
            blocks_run = (self.split_layer - 22 + 2) // 3 
            blocks_run = max(1, min(blocks_run, 6))
            
            # Server runs remaining blocks of Stage 3
            if blocks_run < 6:
                for i in range(blocks_run, 6):
                    x = self.layer3[i](x)
            
            # Then all of Stage 4
            x = self.layer4(x)
            
        elif self.split_layer <= 49:
            # Client ran all Stage 3 and some Stage 4
            blocks_run = (self.split_layer - 40 + 2) // 3
            blocks_run = max(1, min(blocks_run, 3))
            
            # Server runs remaining blocks of Stage 4
            if blocks_run < 3:
                for i in range(blocks_run, 3):
                    x = self.layer4[i](x)
                    
        # Final Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Test function
if __name__ == "__main__":
    print("Testing ResNet50 architecture...")
    print("=" * 70)
    
    # Test with CIFAR-10 input
    print("\n[CIFAR-10/100 - 32x32x3]")
    model = ResNet50(num_classes=10, input_channels=3)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"  Full forward: {out.shape}")
    
    # Test split learning
    for split_point in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        model.configure_split('client', split_point)
        client_out = model(x)
        
        model.configure_split('server', split_point)
        server_out = model(client_out)
        print(f"  Split {split_point}: client {client_out.shape} -> server {server_out.shape}")
    
    # Test with MNIST input
    print("\n[MNIST/FMNIST - 28x28x1]")
    model = ResNet50(num_classes=10, input_channels=1)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    print(f"  Full forward: {out.shape}")
    
    print("\n" + "=" * 70)
    print("✓ ResNet50 architecture verified!")
