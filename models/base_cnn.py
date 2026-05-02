"""Shared custom CNN encoder used by all three detection branches.

Architecture: 5 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool),
channels 64 → 128 → 256 → 512 → 512, followed by global average pooling.
Outputs a 512-dim feature vector. No pretrained weights, no torchvision imports.

Spatial progression for 224×224 input (MaxPool stride=2 each block):
  224 → 112 → 56 → 28 → 14 → 7 → AdaptiveAvgPool(1) → 512-dim

All three unimodal branches (spatial, frequency, temporal) share this exact
architecture, ensuring the ablation study isolates signal contribution rather
than model capacity differences.
"""

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class BaseCNN(nn.Module):
    """5-block custom CNN → 512-dim feature vector.

    Input:  (B, in_channels, 224, 224)
    Output: (B, 512)
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.block1 = _conv_block(in_channels, 64)
        self.block2 = _conv_block(64, 128)
        self.block3 = _conv_block(128, 256)
        self.block4 = _conv_block(256, 512)
        self.block5 = _conv_block(512, 512)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)  # (B, 64,  112, 112)
        x = self.block2(x)  # (B, 128,  56,  56)
        x = self.block3(x)  # (B, 256,  28,  28)
        x = self.block4(x)  # (B, 512,  14,  14)
        x = self.block5(x)  # (B, 512,   7,   7)
        return self.pool(x).flatten(1)  # (B, 512)
