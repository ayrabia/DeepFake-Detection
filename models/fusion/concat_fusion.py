"""Concatenation fusion: three-stream 256-dim projections → 768-dim MLP.

Each branch encoder produces 512-dim features (spatial, frequency) or
512-dim BiLSTM state (temporal). A per-branch linear projection maps each
to exactly 256 dimensions before concatenation so every branch contributes
equally to the 768-dim combined vector. Trained end-to-end jointly.

Inputs:
  spatial_x:  (B, 3, 224, 224)
  freq_x:     (B, 3, 224, 224)
  temporal_x: (B, 16, 3, 224, 224)
Output: (B,) raw logits
"""

import torch
import torch.nn as nn

from models.spatial.spatial_cnn import SpatialCNN
from models.frequency.frequency_cnn import FrequencyCNN
from models.temporal.temporal_lstm import TemporalLSTM

BRANCH_DIM = 512    # raw feature dim from each branch encoder
PROJ_DIM = 256      # projected dim per branch
CONCAT_DIM = PROJ_DIM * 3  # 768


class ConcatFusion(nn.Module):
    def __init__(
        self,
        spatial_model: SpatialCNN,
        frequency_model: FrequencyCNN,
        temporal_model: TemporalLSTM,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.spatial = spatial_model
        self.frequency = frequency_model
        self.temporal = temporal_model

        # Per-branch projection: 512 → 256 so all streams contribute equally
        self.proj_s = nn.Linear(BRANCH_DIM, PROJ_DIM)
        self.proj_f = nn.Linear(BRANCH_DIM, PROJ_DIM)
        self.proj_t = nn.Linear(BRANCH_DIM, PROJ_DIM)

        self.head = nn.Sequential(
            nn.Linear(CONCAT_DIM, hidden_dim),  # 768 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        spatial_x: torch.Tensor,
        freq_x: torch.Tensor,
        temporal_x: torch.Tensor,
    ) -> torch.Tensor:
        f_s = self.proj_s(self.spatial.features(spatial_x))    # (B, 256)
        f_f = self.proj_f(self.frequency.features(freq_x))     # (B, 256)
        f_t = self.proj_t(self.temporal.features(temporal_x))  # (B, 256)
        combined = torch.cat([f_s, f_f, f_t], dim=1)           # (B, 768)
        return self.head(combined).squeeze(1)                   # (B,)
