"""EfficientNetV2 backbone — Phase 3 pretrained comparison only.

Fine-tuned under identical training conditions as the from-scratch SpatialCNN
to isolate the contribution of ImageNet pretraining vs. custom architecture.
Uses timm for the EfficientNetV2 backbone.

NOT used in Phase 1 or Phase 2. Only instantiated in the Phase 3 notebook
alongside Xception for the backbone comparison experiment.

Input:  (B, 3, 224, 224)
Output: (B,) raw logits for BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import timm


class EfficientNetV2CNN(nn.Module):
    def __init__(
        self,
        variant: str = "efficientnetv2_rw_s",
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=0,      # remove classifier head
            drop_rate=dropout,
        )
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)  # (B,)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
