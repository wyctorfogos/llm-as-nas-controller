import torch
import torch.nn as nn
import timm


class LiwTERM(nn.Module):
    """
    Implementação fiel ao paper LiwTERM (SIBGRAPI 2024)
    """

    def __init__(
        self,
        num_classes: int,
        meta_dim: int,
        image_encoder: str = "vit_large_patch16_224",
        pretrained: bool = True,
        unfreeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        # =====================================================
        # 1) Backbone visual (ViT) — FEATURE EXTRACTOR
        # =====================================================
        self.visual = timm.create_model(
            image_encoder,
            pretrained=pretrained,
            num_classes=0 
        )

        self.visual_dim = self.visual.num_features

        if not unfreeze_backbone:
            for p in self.visual.parameters():
                p.requires_grad = False

        # Projeção ViT → 4096 (como no paper)
        self.visual_proj = nn.Sequential(
            nn.Linear(self.visual_dim, 4096),
            nn.LayerNorm(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # =====================================================
        # 2) Projeção dos metadados (OHE)
        # =====================================================
        self.meta_fc = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, 1024),
            nn.ReLU(inplace=True),
        )

        # =====================================================
        # 3) Shallow Lightweight Model (SLM)
        # =====================================================
        concat_dim = 4096 + 1024

        self.slm = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(concat_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes)
        )

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:

        # --- ViT features ---
        v = self.visual.forward_features(image)
        if v.dim() == 3:
            v = v[:, 0]  # CLS token

        v = self.visual_proj(v)

        # --- Metadata features ---
        m = self.meta_fc(metadata)

        # --- Concatenação ---
        x = torch.cat([v, m], dim=1)

        # --- Classificação ---
        logits = self.slm(x)
        return logits
