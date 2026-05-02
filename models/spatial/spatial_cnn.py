"""Spatial (RGB frame) deepfake classifier — Phase 1 unimodal baseline.

Custom 5-block CNN trained from scratch on 224×224 RGB face crops.
No pretrained weights, no torchvision model imports.
Pretrained backbone comparisons (Xception, EfficientNetV2) are Phase 3 only.

Input:  (B, 3, 224, 224)
Output: (B,) raw logits for BCEWithLogitsLoss
"""

import torch
import torch.nn as nn

from models.base_cnn import BaseCNN

CNN_OUT_DIM = 512  # BaseCNN output dimension


class SpatialCNN(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.encoder = BaseCNN(in_channels=3)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(CNN_OUT_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)  # (B,)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """512-dim embedding — consumed by fusion models."""
        return self.encoder(x)
