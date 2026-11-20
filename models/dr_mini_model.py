"""
DR-Mini AI Model Architecture
Lightweight EfficientNet-B0 backbone with 5-class diabetic retinopathy classification head.
Optimised for CPU inference on 8GB RAM PHC laptops.
"""

import torch
import torch.nn as nn
from torchvision import models


class DRMiniAI(nn.Module):
    """
    DR-Mini AI: EfficientNet-B0 backbone with a 5-class DR classification head.

    Architecture:
        EfficientNet-B0 feature extractor (1280-dim output)
        → Dropout(0.3)
        → Linear(1280 → 256) + ReLU
        → Dropout(0.2)
        → Linear(256 → 5)

    Input:  3×224×224 tensor, ImageNet normalization
    Output: 5-class logits [Grade 0 .. Grade 4]
    """

    def __init__(self, num_classes: int = 5):
        super().__init__()
        # Load EfficientNet-B0 backbone without pretrained weights
        # (pretrained weights are baked into the checkpoint during training)
        self.backbone = models.efficientnet_b0(weights=None)

        # Replace the default classifier (1280 → 1000 ImageNet head)
        in_features = self.backbone.classifier[1].in_features  # 1280

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_target_layer_for_cam(self):
        """
        Returns the last convolutional block for Grad-CAM visualization.
        EfficientNet-B0: backbone.features[-1] is the final MBConv block.
        """
        return self.backbone.features[-1]
