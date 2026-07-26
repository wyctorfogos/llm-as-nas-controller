import torch
import torch.nn as nn

class AlteredResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(AlteredResidualBlock, self).__init__()
        # Pré-normalização
        self.norm1 = nn.LayerNorm(dim)
        # Atenção multi-cabeça com um número mais comum de cabeças (por exemplo, 8)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=False)
        self.dropout1 = nn.Dropout(dropout)
        
        # Bloco Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        # Segunda camada de normalização após a soma residual com o bloco FFN
        self.norm2 = nn.LayerNorm(dim)
        
        # Parâmetro de escalonamento para a conexão residual
        self.alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, q, k, v):
        # Pré-normalização da query
        q_norm = self.norm1(q)
        attn_output, _ = self.attn(q_norm, k, v)
        attn_output = self.dropout1(attn_output)
        # Conexão residual com escalonamento
        out1 = q + self.alpha * attn_output
        
        # Passa pelo bloco Feed-Forward
        ffn_output = self.ffn(out1)
        # Segunda conexão residual seguida de normalização
        out2 = self.norm2(out1 + ffn_output)
        return out2