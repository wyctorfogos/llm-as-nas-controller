import torch
import torch.nn as nn

class MetaBlock(nn.Module):
    """
    Metadata Processing Block (MetaBlock)
    Opera em vetores latentes [B, V]
    """
    def __init__(self, V_dim, U_dim):
        super().__init__()

        self.fb = nn.Sequential(
            nn.Linear(U_dim, V_dim),
            nn.LayerNorm(V_dim)
        )

        self.gb = nn.Sequential(
            nn.Linear(U_dim, V_dim),
            nn.LayerNorm(V_dim)
        )

    def forward(self, V, U):
        """
        V: features visuais   -> (B, V_dim)
        U: features metadata  -> (B, U_dim)
        """
        t1 = self.fb(U)  # (B, V_dim)
        t2 = self.gb(U)  # (B, V_dim)

        # Modulação correta (element-wise)
        out = torch.sigmoid(torch.tanh(V * t1) + t2)
        return out
