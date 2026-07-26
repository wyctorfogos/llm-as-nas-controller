import torch
import torch.nn as nn
from models.metablock import MetaBlock


class DynamicCNN(nn.Module):
    def __init__(
        self,
        config: dict,
        in_channels: int = 3,
        num_classes: int = 10,
        attention_mecanism: str = "concatenation",
        text_model_name: str = "one-hot-encoder",
        n: int = 2,
        device: str = "cpu",
        num_heads: int = 8,
        common_dim: int = 512,
        vocab_size: int = 85,
        text_encoder_dim_output: int = 512
    ):
        super().__init__()

        # -------------------------
        # Configurações básicas
        # -------------------------
        self.config = config
        self.device = device
        self.text_encoder_dim_output = text_encoder_dim_output
        self.common_dim = common_dim
        self.attention_mecanism = attention_mecanism
        self.text_model_name = text_model_name
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.n = n

        self.k = int(config.get("kernel_size", 3))
        self.num_layers_text_fc = int(config.get("num_layers_text_fc", 2))
        self.neurons_text = int(config.get("neurons_per_layer_size_of_text_fc", 512))
        self.num_layers_fc = int(config.get("num_layers_fc_module", 2))
        self.neurons_fc = int(config.get("neurons_per_layer_size_of_fc_module", 1024))

        # -------------------------
        # CNN Backbone (mínima mudança)
        # -------------------------
        self.layers = []
        self.in_channels = in_channels

        filters = config.get("filters", [64, 128, 256])

        for out_channels in filters:
            for _ in range(int(config.get("layers_per_block", 2))):
                self.layers.append(
                    nn.Conv2d(
                        self.in_channels,
                        out_channels,
                        kernel_size=self.k,
                        padding=self.k // 2,
                        bias=False
                    )
                )
                # BatchNorm → GroupNorm (mais estável)
                self.layers.append(nn.GroupNorm(8, out_channels))
                self.layers.append(nn.ReLU(inplace=True))
                self.in_channels = out_channels

            if config.get("use_pooling", True):
                self.layers.append(nn.MaxPool2d(2, 2))

        self.conv_block = nn.Sequential(*self.layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_encoder = nn.Sequential(self.conv_block, self.global_pool)

        self.cnn_dim_output = filters[-1]

        # -------------------------
        # Text Encoder (inalterado)
        # -------------------------
        text_layers = [
            nn.Linear(vocab_size, self.neurons_text),
            nn.ReLU()
        ]

        for _ in range(self.num_layers_text_fc):
            text_layers.extend([
                nn.Linear(self.neurons_text, self.neurons_text),
                nn.ReLU()
            ])

        text_layers.append(
            nn.Linear(self.neurons_text, text_encoder_dim_output)
        )

        self.text_fc = nn.Sequential(*text_layers)

        # -------------------------
        # Projeções multimodais
        # -------------------------
        self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)
        self.text_projector = nn.Linear(text_encoder_dim_output, self.common_dim)

        # LayerNorm após projeção (estável entre folds)
        self.ln_img = nn.LayerNorm(self.common_dim)
        self.ln_txt = nn.LayerNorm(self.common_dim)

        # -------------------------
        # Atenções
        # -------------------------
        self.image_self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=self.num_heads,
            batch_first=False
        )

        self.text_self_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=self.num_heads,
            batch_first=False
        )

        self.image_cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=self.num_heads,
            batch_first=False
        )

        self.text_cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_dim,
            num_heads=self.num_heads,
            batch_first=False
        )

        # -------------------------
        # MetaBlock (inalterado)
        # -------------------------
        # Bloco do Metablock, caso queira usar
        self.meta_block = MetaBlock(V_dim=self.common_dim if self.attention_mecanism in ["att-intramodal+residual+cross-attention-metadados+metablock"] else self.cnn_dim_output,
            U_dim=self.common_dim if self.attention_mecanism in ["att-intramodal+residual+cross-attention-metadados+metablock"] else self.text_encoder_dim_output
        )

        # -------------------------
        # Classificador
        # -------------------------
        self.fc_fusion = self._build_fc_fusion(
            n=1 if attention_mecanism == "no-metadata" else self.n
        )

        self.fc_visual_only = nn.Linear(self.cnn_dim_output, self.num_classes)

    # ======================================================
    # MLP de fusão (BatchNorm → LayerNorm)
    # ======================================================
    def _build_fc_fusion(self, n=1):
        layers = [
            nn.Linear(self.common_dim * n, self.neurons_fc),
            nn.LayerNorm(self.neurons_fc),
            nn.ReLU(),
            nn.Dropout(0.3)
        ]

        for _ in range(self.num_layers_fc):
            layers.extend([
                nn.Linear(self.neurons_fc, self.neurons_fc),
                nn.ReLU()
            ])

        layers.append(nn.Linear(self.neurons_fc, self.num_classes))
        return nn.Sequential(*layers)

    # ======================================================
    # Forward
    # ======================================================
    def forward(self, image, text_metadata):
        image = image.to(self.device)

        # ---------- Image ----------
        img_feat = self.image_encoder(image)           # (B, C, 1, 1)
        img_feat = img_feat.view(img_feat.size(0), -1) # (B, C)

        img_proj = self.image_projector(img_feat)
        img_proj = self.ln_img(img_proj)

        img_seq = img_proj.unsqueeze(0)  # (1, B, D)

        # ---------- Text ----------
        txt_feat = self.text_fc(text_metadata)         # (B, D_txt)
        txt_proj = self.text_projector(txt_feat)
        txt_proj = self.ln_txt(txt_proj)

        txt_seq = txt_proj.unsqueeze(0)  # (1, B, D)

        # ---------- Atenções ----------
        img_att, _ = self.image_self_attention(img_seq, img_seq, img_seq)
        txt_att, _ = self.text_self_attention(txt_seq, txt_seq, txt_seq)

        img_cross, _ = self.image_cross_attention(img_att, txt_att, txt_att)
        txt_cross, _ = self.text_cross_attention(txt_att, img_att, img_att)

        img_pooled = img_cross.squeeze(0)
        txt_pooled = txt_cross.squeeze(0)

        # ---------- Saídas ----------
        if self.attention_mecanism == "no-metadata":
            return self.fc_fusion(img_proj)

        elif self.attention_mecanism == "concatenation":
            return self.fc_fusion(torch.cat([img_pooled, txt_pooled], dim=1))

        elif self.attention_mecanism == "crossattention":
            return self.fc_fusion(torch.cat([img_pooled, txt_pooled], dim=1))

        elif self.attention_mecanism == "metablock":
            meta_features = self.meta_block(
                img_feat,      
                txt_feat
            )
            return self.fc_visual_only(meta_features)

        else:
            raise ValueError(f"Attention mechanism '{self.attention_mecanism}' not supported.")
