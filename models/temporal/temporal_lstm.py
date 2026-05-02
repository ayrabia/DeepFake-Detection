"""Temporal (frame sequence) deepfake classifier — Phase 1 unimodal baseline.

Architecture: shared BaseCNN frame encoder → bidirectional LSTM → binary head.
16 frames sampled per clip (proposal spec, sequence_length=16).
The CNN encoder is the same BaseCNN architecture as the spatial/frequency branches.

Input:  (B, 16, 3, 224, 224)
Output: (B,) raw logits for BCEWithLogitsLoss

Batch sizes (from compute plan):
  A100: 16 clips  |  T4: 8 clips
"""

import torch
import torch.nn as nn

from models.base_cnn import BaseCNN

CNN_OUT_DIM = 512     # BaseCNN output
LSTM_HIDDEN = 256     # BiLSTM hidden size per direction → 512 total


class TemporalLSTM(nn.Module):
    def __init__(
        self,
        sequence_length: int = 16,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.sequence_length = sequence_length

        # Shared weights: same CNN encodes every frame in the sequence
        self.frame_encoder = BaseCNN(in_channels=3)

        self.lstm = nn.LSTM(
            input_size=CNN_OUT_DIM,      # 512 from BaseCNN
            hidden_size=LSTM_HIDDEN,     # 256 per direction
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = LSTM_HIDDEN * 2   # 512 (bidirectional)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        frame_feats = self.frame_encoder(x.view(B * T, C, H, W)).view(B, T, -1)
        lstm_out, _ = self.lstm(frame_feats)             # (B, T, 512)
        return self.head(lstm_out[:, -1, :]).squeeze(1)  # (B,)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """BiLSTM last-step hidden state (B, 512) — for late/concat fusion."""
        B, T, C, H, W = x.shape
        frame_feats = self.frame_encoder(x.view(B * T, C, H, W)).view(B, T, -1)
        lstm_out, _ = self.lstm(frame_feats)
        return lstm_out[:, -1, :]  # (B, 512)

    def frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per-frame CNN embeddings (B, T, 512) — for cross-attention fusion.

        Returns pre-LSTM representations so attention can focus on individual
        frames rather than the collapsed sequence state.
        """
        B, T, C, H, W = x.shape
        return self.frame_encoder(x.view(B * T, C, H, W)).view(B, T, -1)
