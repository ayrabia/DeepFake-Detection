"""Frequency (FFT magnitude spectrum) deepfake classifier — Phase 1 unimodal baseline.

Identical architecture to SpatialCNN — same BaseCNN trained independently on
FFT magnitude images. Using the exact same architecture for both branches
ensures the ablation study isolates signal contribution, not model capacity.

FFT images are saved as grayscale JPEGs but loaded as 3-channel by OpenCV,
so in_channels=3 is used throughout.

Input:  (B, 3, 224, 224)
Output: (B,) raw logits for BCEWithLogitsLoss
"""

import torch
import torch.nn as nn

from models.base_cnn import BaseCNN

CNN_OUT_DIM = 512


class FrequencyCNN(nn.Module):
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
