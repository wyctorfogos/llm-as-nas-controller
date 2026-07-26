#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: André Pacheco
E-mail: pacheco.comp@gmail.com

This file implements the Context Guided Cell (GCell)
and a full MetaNet + ResNet-50 model.

Paper:
Fusing Metadata and Dermoscopy Images for Skin Disease Diagnosis
IEEE Journal of Biomedical and Health Informatics, 2020
https://ieeexplore.ieee.org/document/9098645
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================
# MetaNet block (Context Guided Cell)
# =====================================================
class MetaNet(nn.Module):
    """
    Metadata-driven channel attention (MetaNet / GCell)

    metadata (B, meta_dim)  →  (B, C, 1, 1)
    feat_maps (B, C, H, W)  →  gated feature maps
    """

    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()
        self.metanet = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat_maps: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        feat_maps: (B, C, H, W)
        metadata:  (B, meta_dim)
        """
        m = metadata.unsqueeze(-1).unsqueeze(-1)  # (B, meta_dim, 1, 1)
        attn = self.metanet(m)                    # (B, C, 1, 1)
        return feat_maps * attn


# =====================================================
# MetaNet + ResNet-50 model
# =====================================================
class MetaNetModel(nn.Module):
    """
    MetaNet + ResNet-50 (faithful to IEEE JBHI paper)
    """

    def __init__(
        self,
        meta_dim: int,
        num_classes: int = 6,
        dropout_fraction: float = 0.3,
        image_encoder: str = "resnet50",
        pretrained: bool = True,
        unfreeze_weights: bool = False
    ):
        super().__init__()

        self.meta_dim = meta_dim
        self.num_classes = num_classes
        self.dropout_fraction = dropout_fraction
        self.image_encoder = image_encoder
        self.pretrained = pretrained
        self.unfreeze_weights = unfreeze_weights

        # =====================================================
        # 1) CNN backbone (conv features only)
        # =====================================================
        self.backbone = timm.create_model(
            self.image_encoder,
            pretrained=self.pretrained,
            num_classes=0,      # ❗ remove FC
            global_pool=""      # ❗ remove GAP → retorna (B,C,H,W)
        )

        self.feat_dim = self.backbone.num_features  # 2048 for resnet50

        if not self.unfreeze_weights:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # =====================================================
        # 2) MetaNet attention (metadata → channel gates)
        # =====================================================
        self.metanet = MetaNet(
            in_channels=self.meta_dim,
            middle_channels=128,
            out_channels=self.feat_dim
        )

        # =====================================================
        # 3) Classifier (after GAP)
        # =====================================================
        self.classifier = self.fc_mlp_module(self.feat_dim)

    # -----------------------------------------------------
    # MLP classifier (stronger than single FC)
    # -----------------------------------------------------
    def fc_mlp_module(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_fraction),

            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_fraction),

            nn.Linear(input_dim // 2, self.num_classes)
        )

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        image:    (B, 3, 224, 224)
        metadata: (B, meta_dim)
        """

        # 1) CNN feature maps
        feat_maps = self.backbone(image)          # (B, 2048, H, W)

        # 2) Metadata-guided channel attention
        feat_maps = self.metanet(feat_maps, metadata)

        # 3) Global Average Pooling
        pooled = F.adaptive_avg_pool2d(feat_maps, 1).flatten(1)  # (B, 2048)

        # 4) Classification
        logits = self.classifier(pooled)
        return logits
