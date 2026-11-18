"""
Diabetic Retinopathy Classification Deep Neural Network Architecture
Based on ResNet-50 backbone with specialized classification head for
5-class ICDR Diabetic Retinopathy Severity Grading.
"""

import torch
import torch.nn as nn
from torchvision import models

class DRResNet50(nn.Module):
    """
    ResNet-50 architecture adapted for 5-class Diabetic Retinopathy Grading:
    Grade 0: No DR
    Grade 1: Mild NPDR
    Grade 2: Moderate NPDR
    Grade 3: Severe NPDR
    Grade 4: Proliferative DR
    """
    def __init__(self, num_classes=5, pretrained_backbone=False):
        super(DRResNet50, self).__init__()
        
        # Load torchvision ResNet50
        weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = models.resnet50(weights=weights)
        
        # Target layer for Grad-CAM explainability is layer4[-1]
        self.target_layer = self.backbone.layer4[-1]
        
        # In-features of ResNet-50 standard final layer is 2048
        in_features = self.backbone.fc.in_features
        
        # Clinically calibrated multi-layer classification head with dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_target_layer_for_cam(self):
        """Returns the final convolutional bottleneck block for Grad-CAM."""
        return self.backbone.layer4[-1]
