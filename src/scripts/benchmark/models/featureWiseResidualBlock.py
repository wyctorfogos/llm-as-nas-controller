import torch
import torch.nn as nn

class FeatureModulation(nn.Module):
    def __init__(self, dim, meta_dim):
        super().__init__()
        self.scale = nn.Linear(meta_dim, dim)
        self.shift = nn.Linear(meta_dim, dim)

    def forward(self, visual_feats, metadata):
        """
        visual_feats: (seq_len, batch, dim)
        metadata: (batch, meta_dim)
        """
        scale = self.scale(metadata).unsqueeze(0)  # (1, batch, dim)
        shift = self.shift(metadata).unsqueeze(0)  # (1, batch, dim)
        return visual_feats * scale + shift
