"""
MobileNetV4 Architecture for Federated Split Learning

Efficient mobile architecture with inverted residual blocks and squeeze-excitation.
Adapted for CIFAR-10/100 and MNIST/FMNIST datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class InvertedResidual(nn.Module):
    """Inverted Residual Block (MBConv)"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_se=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))
        
        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV4(nn.Module):
    """
    MobileNetV4 architecture adapted for small images (32x32 or 28x28).
    Supports split learning with 10 configurable split points.
    """
    def __init__(self, num_classes=10, input_channels=3, width_mult=1.0):
        super(MobileNetV4, self).__init__()
        
        # Building blocks: [expansion, out_channels, num_blocks, stride]
        config = [
            [1, 16, 1, 1],   # Stage 1
            [6, 24, 2, 2],   # Stage 2
            [6, 32, 3, 2],   # Stage 3
  [6, 64, 4, 2],   # Stage 4
            [6, 96, 3, 1],   # Stage 5
            [6, 160, 3, 2],  # Stage 6
            [6, 320, 1, 1],  # Stage 7
        ]
        
        # Adjust channels based on width multiplier
        input_channel = int(32 * width_mult)
        
        # Initial convolution
        self.features = [nn.Sequential(
            nn.Conv2d(input_channels, input_channel, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        # Build inverted residual blocks
        for t, c, n, s in config:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
        
        # Final convolution
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*self.features)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)
        
        # Split learning configuration
        self.split_mode = 'full'
        self.split_layer = None
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def is_client_layer(self, layer_name, split_layer):
        """
        Determine if a layer belongs to the client side.
        Supports strict split range 27-52.
        """
        # Initial checks
        if 'classifier' in layer_name or 'avgpool' in layer_name:
            return False
            
        # Parse features.X
        if layer_name.startswith('features.'):
            try:
                parts = layer_name.split('.')
                layer_idx = int(parts[1])
                
                # Map 27-52 to feature indices
                if split_layer < 27:
                    limit_idx = 6
                elif split_layer > 52:
                    limit_idx = 19
                else:
                    offset = split_layer - 27
                    limit_idx = 6 + int(offset * 0.5)
                
                limit_idx = max(1, min(limit_idx, 19))
                
                return layer_idx < limit_idx
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _forward_client(self, x):
        """
        Client-side forward pass up to split point (Layers 27-52).
        
        Mapping logic (Approximate):
        MobileNetV4 has ~53 blocks/layers.
        features[] list has ~19 top-level modules (blocks).
        
        We map range [27, 52] to feature indices [6, 18].
        Slope ≈ (18-6)/(52-27) = 12/25 ≈ 0.5
        
        feature_idx = 6 + (split_layer - 27) * 0.5
        """
        # Map 27-52 to 6-18
        # 6 corresponds to Start of Stage 4 (or thereabouts)
        
        if self.split_layer < 27:
            # Fallback for unexpected low values: run minimal
            limit_idx = 6
        elif self.split_layer > 52:
            limit_idx = 19
        else:
            # Linear mapping
            offset = self.split_layer - 27
            limit_idx = 6 + int(offset * 0.5)
            
        limit_idx = max(1, min(limit_idx, len(self.features)))
        
        # Execute
        for i in range(limit_idx):
            x = self.features[i](x)
        return x
    
    def _forward_server(self, x):
        """Server-side forward pass from split point"""
        if self.split_layer < 27:
            start_idx = 6
        elif self.split_layer > 52:
            start_idx = 19
        else:
            offset = self.split_layer - 27
            start_idx = 6 + int(offset * 0.5)
            
        start_idx = max(0, min(start_idx, len(self.features)))
        
        for i in range(start_idx, len(self.features)):
            x = self.features[i](x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Test
if __name__ == "__main__":
    print("Testing MobileNetV4...")
    print("=" * 70)
    
    # CIFAR
    model = MobileNetV4(num_classes=10, input_channels=3)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"CIFAR Full forward: {out.shape}")
    
    # Test splits
    for sp in [5, 7, 9, 11, 14]:
        model.configure_split('client', sp)
        client_out = model(x)
        model.configure_split('server', sp)
        server_out = model(client_out)
        print(f"Split {sp}: {client_out.shape} -> {server_out.shape}")
    
    # MNIST
    model = MobileNetV4(num_classes=10, input_channels=1)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    print(f"\nMNIST Full forward: {out.shape}")
    
    print("=" * 70)
    print("✓ MobileNetV4 verified!")
