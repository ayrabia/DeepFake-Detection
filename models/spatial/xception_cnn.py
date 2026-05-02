"""Xception backbone — Phase 3 pretrained SOTA baseline.

Xception is the published SOTA baseline for FaceForensics++ (Rossler et al.,
ICCV 2019). Evaluated under identical conditions to benchmark the from-scratch
custom CNN. Uses torchvision's feature_extraction to pull the final pooled
features for optional fusion use.

Input:  (B, 3, 299, 299)  — Xception expects 299×299 not 224×224
Output: (B,) raw logits
"""

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class XceptionCNN(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm is required for XceptionCNN. Install with: pip install timm"
            )
        # timm provides Xception with pretrained ImageNet weights
        self.backbone = timm.create_model(
            "xception", pretrained=pretrained, num_classes=0
        )  # num_classes=0 → returns features, not logits
        feat_dim = self.backbone.num_features  # 2048
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)  # (B,)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
