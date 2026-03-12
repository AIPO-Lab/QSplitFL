"""
ConvNeXt Architecture for Federated Split Learning

Modern CNN inspired by vision transformers with large kernels and LayerNorm.
Adapted for CIFAR-10/100 and MNIST/FMNIST datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with large kernel depthwise conv"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        x = input + x
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny architecture adapted for small images (32x32 or 28x28).
    Supports split learning with 10 configurable split points.
    
    Block configuration: [3, 3, 9, 3] for Tiny variant
    """
    def __init__(self, num_classes=10, input_channels=3, depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768]):
        super().__init__()
        
        # Stem - patchify with 4x4 conv (stride 4 -> 2 for small images)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers between stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        # 4 feature resolution stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        # Head
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        
        # Split learning configuration
        self.split_mode = 'full'
        self.split_layer = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def is_client_layer(self, layer_name, split_layer):
        """
        Determine if a layer belongs to the client side.
        Supports strict split range 30-58.
        """
        # Head and norm are always server for supported splits
        if 'head' in layer_name or 'norm' in layer_name:
            if 'stages' not in layer_name and 'downsample' not in layer_name:
                return False

        # Determine how many blocks in stage 2 and 3 are client
        client_s2_blocks = 9
        client_s3_blocks = 0
        
        if split_layer < 37:
            offset = split_layer - 30
            client_s2_blocks = min(9, 4 + offset)
        else:
            client_s2_blocks = 9
            
        if split_layer >= 37:
            if split_layer <= 43: client_s3_blocks = 1
            elif split_layer <= 50: client_s3_blocks = 2
            else: client_s3_blocks = 3
            
        if layer_name.startswith('downsample_layers.'):
            try:
                idx = int(layer_name.split('.')[1])
                # 0, 1 always client (Stage 0, 1 are client in range 30-58)
                if idx <= 1: return True
                # 2 is before stage 2, always client
                if idx == 2: return True 
                
                # 3 is before stage 3. 
                # If we are splitting in Stage 2 (split < 37), then downsample 3 is server.
                # If we are splitting in Stage 3 (split >= 37), then downsample 3 is client.
                if idx == 3:
                     return split_layer >= 37
                return False
            except: return False
            
        if layer_name.startswith('stages.'):
            try:
                parts = layer_name.split('.')
                stage_idx = int(parts[1])
                block_idx = int(parts[2]) if len(parts) > 2 else -1
                
                if stage_idx <= 1: return True
                
                if stage_idx == 2:
                    return block_idx < client_s2_blocks
                    
                if stage_idx == 3:
                    return block_idx < client_s3_blocks
                    
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
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        x = x.mean([-2, -1])  # Global average pooling
        x = self.norm(x)
        x = self.head(x)
        return x
    
    def _forward_client(self, x):
        """
        Client-side forward pass up to split point (Layers 30-58).
        
        Mapping logic:
        ConvNeXt has 4 Stages: [3, 3, 9, 3] blocks.
        Approx depth:
        - Stage 0: Layers 1-4
        - Stage 1: Layers 5-8
        - Stage 2: Layers 9-36 (Start 30 is late Stage 2)
        - Stage 3: Layers 37-58 (Start 37 is Stage 3)
        """
        # Always run Stage 0 and Stage 1 (and downsamples)
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        x = self.downsample_layers[2](x) # Into Stage 2
        
        # Stage 2 (9 blocks). Indices 0..8
        # Range 30-36 falls in Stage 2.
        # Range 37-58 falls in Stage 3.
        
        # Logic:
        # If split < 37: we are in Stage 2.
        # 30 -> Block 6
        # 31 -> Block 7
        # 32 -> Block 8 (End Stage 2)
        # Wait, let's distribute 30-36 across end of Stage 2?
        # Let's say 30 corresponds to Block 3 of Stage 2?
        # Let's map 30 to Stage 2 Block 6.
        
        client_s2_blocks = 9 # Run all by default if split >= 37
        client_s3_blocks = 0
        
        if self.split_layer < 37:
            # Map 30-36 to Stage 2 blocks
            # 30 -> 6, 31 -> 7, 32 -> 8
            # This is a bit tight. Let's start earlier in Stage 2 if needed?
            # User strict range is 30-58.
            # Let's map linearly 30 to block 3, 36 to block 8?
            # 30 -> 4, 31 -> 5, 32 -> 6, 33 -> 7, 34 -> 8 ?
            
            offset = self.split_layer - 30
            client_s2_blocks = min(9, 4 + offset) # Start at block 4
            
        else:
            # split >= 37. Run all Stage 2.
            client_s2_blocks = 9
            
            # Map 37-58 to Stage 3 (3 blocks).
            # This range is wide (20 pts) for only 3 blocks.
            # But ConvNeXt blocks are "deep" (depth 3).
            # 58 / 3 approx 19 blocks total.
            # We have 3+3+9+3 = 18 blocks.
            # So 30 is indeed deep.
            
            # We will just run full Stage 2.
            pass
            
        # Execute Stage 2
        for i in range(client_s2_blocks):
            x = self.stages[2][i](x)
            
        if self.split_layer < 37:
            return x
            
        # Stage 3 logic
        # Downsample 3
        x = self.downsample_layers[3](x)
        
        if self.split_layer >= 37:
            # Map 37-58 to Stage 3 blocks (0, 1, 2)
            # 37-43 -> Block 0
            # 44-50 -> Block 1
            # 51-58 -> Block 2
            
            if self.split_layer <= 43:
                client_s3_blocks = 1
            elif self.split_layer <= 50:
                client_s3_blocks = 2
            else:
                client_s3_blocks = 3
                
            for i in range(client_s3_blocks):
                x = self.stages[3][i](x)
                
        return x
    
    def _forward_server(self, x):
        """Server-side forward pass"""
        
        # Recover context
        client_s2_blocks = 9
        client_s3_blocks = 0
        
        if self.split_layer < 37:
            offset = self.split_layer - 30
            client_s2_blocks = min(9, 4 + offset)
            
            # Server finishes Stage 2
            for i in range(client_s2_blocks, 9):
                x = self.stages[2][i](x)
                
            # Server does Stage 3 full
            x = self.downsample_layers[3](x)
            x = self.stages[3](x)
            
        else:
            # Stage 2 was done
            # Stage 3 logic
            if self.split_layer <= 43:
                client_s3_blocks = 1
            elif self.split_layer <= 50:
                client_s3_blocks = 2
            else:
                client_s3_blocks = 3
            
            # Server finishes Stage 3
            for i in range(client_s3_blocks, 3):
                x = self.stages[3][i](x)
                
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x


# Test
if __name__ == "__main__":
    print("Testing ConvNeXt...")
    print("=" * 70)
    
    # CIFAR
    model = ConvNeXt(num_classes=10, input_channels=3)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"CIFAR Full forward: {out.shape}")
    
    # Test all splits
    for sp in range(5, 15):
        model.configure_split('client', sp)
        client_out = model(x)
        model.configure_split('server', sp)
        server_out = model(client_out)
        print(f"Split {sp}: {client_out.shape} -> {server_out.shape}")
    
    # MNIST
    model = ConvNeXt(num_classes=10, input_channels=1)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    print(f"\nMNIST Full forward: {out.shape}")
    
    print("=" * 70)
    print("✓ ConvNeXt verified!")
