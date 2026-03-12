"""
ResNet50 Architecture with Split Learning Support

Adapted from the QSplitFL complete_rl_implementation for the ablation study.

Split point range: layers 25–49 (L=50, min_split = ⌈50/2⌉ = 25)
Layer counting:
  conv1 + bn1 + relu + maxpool            →  initial block
  layer1 (3 Bottleneck blocks, 3 layers each) →  layers  1–9  (always client)
  layer2 (4 Bottleneck blocks)               →  layers 10–21  (always client)
  layer3 (6 Bottleneck blocks)               →  layers 22–40  (split range)
  layer4 (3 Bottleneck blocks)               →  layers 41–49  (split range)
  avgpool + fc                               →  always server (output head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """Standard ResNet50 bottleneck block: 1×1 → 3×3 → 1×1."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet50(nn.Module):
    """
    ResNet50 adapted for small images (32×32 CIFAR or 28×28 MNIST).
    Supports split learning with configurable split point in range 25–49.
    """

    TOTAL_LAYERS = 50
    MIN_SPLIT    = 25   # ⌈50/2⌉
    MAX_SPLIT    = 49
    ACTION_SIZE  = MAX_SPLIT - MIN_SPLIT + 1   # 25 discrete actions

    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.in_channels = 64

        # Stem — use 3×3 conv (stride=1) for small images instead of 7×7 stride-2
        self.conv1   = nn.Conv2d(input_channels, 64, kernel_size=3,
                                 stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet stages with bottleneck counts [3, 4, 6, 3]
        self.layer1  = self._make_layer(64,  3, stride=1)
        self.layer2  = self._make_layer(128, 4, stride=2)
        self.layer3  = self._make_layer(256, 6, stride=2)
        self.layer4  = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Split learning state
        self.split_mode  = 'full'    # 'full' | 'client' | 'server'
        self.split_layer = None

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def _stage3_blocks(split_layer: int) -> int:
        """How many layer3 blocks does the client run for a given split_layer?"""
        if split_layer <= 22:
            return 0
        if split_layer >= 40:
            return 6
        return max(1, min(6, (split_layer - 22 + 2) // 3))

    @staticmethod
    def _stage4_blocks(split_layer: int) -> int:
        """How many layer4 blocks does the client run for a given split_layer?"""
        if split_layer <= 40:
            return 0
        if split_layer >= 49:
            return 3
        return max(1, min(3, (split_layer - 40 + 2) // 3))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure_split(self, mode: str, split_layer: int):
        """Set split learning mode before a forward pass."""
        assert mode in ('full', 'client', 'server'), f"Unknown mode: {mode}"
        self.split_mode  = mode
        self.split_layer = split_layer

    def is_client_layer(self, layer_name: str, split_layer: int) -> bool:
        """
        Return True if `layer_name` (state_dict key) belongs to the client
        side for the given split_layer.
        """
        # Output head always stays on the server
        if 'fc' in layer_name or 'avgpool' in layer_name:
            return False
        # Stem and stages 1–2 always go to the client
        if any(layer_name.startswith(p) for p in
               ('conv1', 'bn1', 'layer1', 'layer2')):
            return True
        # Stage 3 — partial split
        if layer_name.startswith('layer3'):
            parts = layer_name.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                return int(parts[1]) < self._stage3_blocks(split_layer)
            return False
        # Stage 4 — partial split
        if layer_name.startswith('layer4'):
            parts = layer_name.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                n3 = self._stage3_blocks(split_layer)
                if n3 < 6:          # client didn't finish stage 3 yet
                    return False
                return int(parts[1]) < self._stage4_blocks(split_layer)
            return False
        return False

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x):
        if self.split_mode == 'full':
            return self._forward_full(x)
        elif self.split_mode == 'client':
            return self._forward_client(x)
        elif self.split_mode == 'server':
            return self._forward_server(x)
        raise ValueError(f"Unknown split_mode: {self.split_mode}")

    def _forward_full(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))

    def _forward_client(self, x):
        """Forward through layers 1..split_layer → smashed data."""
        # Stem + stages 1–2 (always client)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        n3 = self._stage3_blocks(self.split_layer)
        for i in range(n3):
            x = self.layer3[i](x)

        # If client finished all of stage 3, run some stage 4 blocks
        if n3 == 6:
            n4 = self._stage4_blocks(self.split_layer)
            for i in range(n4):
                x = self.layer4[i](x)

        return x

    def _forward_server(self, x):
        """Forward through layers split_layer+1..L → logits."""
        n3 = self._stage3_blocks(self.split_layer)
        n4 = self._stage4_blocks(self.split_layer)

        # Remaining stage 3 blocks
        if n3 < 6:
            for i in range(n3, 6):
                x = self.layer3[i](x)
            x = self.layer4(x)
        else:
            # Client finished all of stage 3; run remaining stage 4 blocks
            for i in range(n4, 3):
                x = self.layer4[i](x)

        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing ResNet50 split learning…")
    model = ResNet50(num_classes=10, input_channels=3)
    x = torch.randn(4, 3, 32, 32)

    # Full forward
    out_full = model(x)
    print(f"Full forward: {out_full.shape}")

    # Split forward for a range of split points
    for sp in [25, 30, 35, 40, 45, 49]:
        model.configure_split('client', sp)
        smashed = model(x)
        model.configure_split('server', sp)
        out_split = model(smashed)
        print(f"  split={sp}: smashed={smashed.shape}  logits={out_split.shape}")

    print("✓ ResNet50 OK")
