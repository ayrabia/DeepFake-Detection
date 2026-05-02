"""Late fusion: average sigmoid outputs from the three independent unimodal models.

No additional training required. The three baselines are trained independently
and their probabilities are averaged at inference time. Simple but often
competitive baseline for the ablation study.

Inputs:
  spatial_x:  (B, 3, 224, 224)
  freq_x:     (B, 3, 224, 224)
  temporal_x: (B, T, 3, 224, 224)
Output: (B,) averaged probability in [0, 1]  (fake probability)
"""

import torch
import torch.nn as nn

from models.spatial.spatial_cnn import SpatialCNN
from models.frequency.frequency_cnn import FrequencyCNN
from models.temporal.temporal_lstm import TemporalLSTM


class LateFusion(nn.Module):
    def __init__(
        self,
        spatial_model: SpatialCNN,
        frequency_model: FrequencyCNN,
        temporal_model: TemporalLSTM,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        super().__init__()
        self.spatial = spatial_model
        self.frequency = frequency_model
        self.temporal = temporal_model
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", w / w.sum())

    def forward(
        self,
        spatial_x: torch.Tensor,
        freq_x: torch.Tensor,
        temporal_x: torch.Tensor,
    ) -> torch.Tensor:
        p_s = torch.sigmoid(self.spatial(spatial_x))
        p_f = torch.sigmoid(self.frequency(freq_x))
        p_t = torch.sigmoid(self.temporal(temporal_x))
        return (
            self.weights[0] * p_s
            + self.weights[1] * p_f
            + self.weights[2] * p_t
        )  # (B,) — probability in [0, 1]
