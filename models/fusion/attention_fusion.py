"""Cross-attention fusion — primary research contribution (Phase 2).

Three pairwise cross-attention operations where each stream attends over
one other stream, then the three attended outputs are concatenated and
classified by an MLP head. Trained end-to-end jointly.

Attention flow (per proposal):
  1. spatial   → attends over frequency      (spatial   as Q, freq     as K/V)
  2. frequency → attends over temporal       (frequency as Q, temporal as K/V)
  3. temporal  → attends over spatial        (temporal  as Q, spatial  as K/V)

All branch features are projected to ATTN_DIM (256) before attention so
that MHA embed_dim is consistent across operations regardless of raw feature sizes.

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

BRANCH_DIM = 512    # raw feature dim from each branch
ATTN_DIM = 256      # projected dim for attention operations
NUM_HEADS = 4       # must divide ATTN_DIM evenly
CONCAT_DIM = ATTN_DIM * 3  # 768


class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        spatial_model: SpatialCNN,
        frequency_model: FrequencyCNN,
        temporal_model: TemporalLSTM,
        num_heads: int = NUM_HEADS,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.spatial = spatial_model
        self.frequency = frequency_model
        self.temporal = temporal_model

        # Project each branch to ATTN_DIM before attention
        self.proj_s = nn.Linear(BRANCH_DIM, ATTN_DIM)
        self.proj_f = nn.Linear(BRANCH_DIM, ATTN_DIM)
        self.proj_t = nn.Linear(BRANCH_DIM, ATTN_DIM)  # applied per-frame

        # Attention 1: spatial (Q) attends over frequency (K/V)
        self.attn_sf = nn.MultiheadAttention(
            embed_dim=ATTN_DIM, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_sf = nn.LayerNorm(ATTN_DIM)

        # Attention 2: frequency (Q) attends over temporal frames (K/V)
        self.attn_ft = nn.MultiheadAttention(
            embed_dim=ATTN_DIM, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ft = nn.LayerNorm(ATTN_DIM)

        # Attention 3: temporal frames (Q) attend over spatial (K/V)
        self.attn_ts = nn.MultiheadAttention(
            embed_dim=ATTN_DIM, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ts = nn.LayerNorm(ATTN_DIM)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(CONCAT_DIM, ATTN_DIM),  # 768 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(ATTN_DIM, 1),
        )

    def forward(
        self,
        spatial_x: torch.Tensor,
        freq_x: torch.Tensor,
        temporal_x: torch.Tensor,
    ) -> torch.Tensor:
        # Project all branches to ATTN_DIM
        # Spatial and frequency: (B, 512) → (B, 256) → (B, 1, 256) for MHA
        s = self.proj_s(self.spatial.features(spatial_x)).unsqueeze(1)   # (B, 1, 256)
        f = self.proj_f(self.frequency.features(freq_x)).unsqueeze(1)    # (B, 1, 256)

        # Temporal: (B, T, 512) → (B, T, 256) using per-frame CNN embeddings
        t_frames = self.temporal.frame_features(temporal_x)              # (B, T, 512)
        t = self.proj_t(t_frames.view(-1, BRANCH_DIM)).view(
            t_frames.size(0), t_frames.size(1), ATTN_DIM
        )                                                                 # (B, T, 256)

        # ── Attention 1: spatial queries frequency ────────────────────────
        s_att, _ = self.attn_sf(query=s, key=f, value=f)
        s_out = self.norm_sf((s + s_att).squeeze(1))                     # (B, 256)

        # ── Attention 2: frequency queries temporal frames ────────────────
        f_att, _ = self.attn_ft(query=f, key=t, value=t)
        f_out = self.norm_ft((f + f_att).squeeze(1))                     # (B, 256)

        # ── Attention 3: temporal frames query spatial ────────────────────
        t_att, _ = self.attn_ts(query=t, key=s, value=s)
        t_out = self.norm_ts((t + t_att).mean(dim=1))                    # (B, 256) mean over T

        # Concatenate attended representations and classify
        combined = torch.cat([s_out, f_out, t_out], dim=1)               # (B, 768)
        return self.head(combined).squeeze(1)                             # (B,)
