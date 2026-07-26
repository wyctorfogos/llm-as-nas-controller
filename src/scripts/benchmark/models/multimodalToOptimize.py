import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loadImageModelClassifier import loadModels
class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device, cnn_model_name, text_model_name, vocab_size=85, attention_mecanism="combined",
                 text_fc_config=None, num_heads=8, fc_fusion_config=None):
        super(MultimodalModel, self).__init__()
        
        # Dimensões do modelo
        self.common_dim = 512
        self.text_encoder_dim_output = 512
        self.device = device
        self.cnn_model_name = cnn_model_name
        self.text_model_name = text_model_name
        self.attention_mecanism = attention_mecanism
        self.num_heads = num_heads  # número de cabeças da atenção multihead

        # -------------------------
        # 1) Image Encoder
        # -------------------------
        self.image_encoder, self.cnn_dim_output = loadModels.loadModelImageEncoder(
            self.cnn_model_name, self.common_dim
        )
        
        self.image_projector = nn.Linear(self.cnn_dim_output, self.common_dim)

        # -------------------------
        # 2) Text Encoder
        # -------------------------
        if self.text_model_name == "one-hot-encoder":
            layers = []
            hidden_sizes = text_fc_config.get('hidden_sizes', [256, 512])
            dropout = text_fc_config.get('dropout', 0.3)
            input_dim = vocab_size

            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_size
            layers.append(nn.Linear(input_dim, self.text_encoder_dim_output))
            self.text_fc = nn.Sequential(*layers)
        else:
            # Implementação para outros encoders de texto
            pass
        
        # Projeção final p/ espaço comum
        self.text_projector = nn.Linear(self.text_encoder_dim_output, self.common_dim)

        # -------------------------
        # 3) Atenções
        # -------------------------
        self.image_self_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=num_heads, batch_first=False)
        self.text_self_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=num_heads, batch_first=False)
        
        self.image_cross_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=num_heads, batch_first=False)
        self.text_cross_attention = nn.MultiheadAttention(embed_dim=self.common_dim, num_heads=num_heads, batch_first=False)

        # -------------------------
        # 4) Fusão Final
        # -------------------------
        layers = []
        hidden_sizes = fc_fusion_config.get('hidden_sizes', [1024, 512])
        dropout = fc_fusion_config.get('dropout', 0.3)
        input_dim = self.common_dim * 2

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, num_classes))
        layers.append(nn.Softmax(dim=1))
        self.fc_fusion = nn.Sequential(*layers)

    def forward(self, image, text_metadata):
        """
        image: tensor de imagens (batch, C, H, W) se CNN 
               ou lista PIL se ViTFeatureExtractor
        text_metadata: dicionário c/ 'input_ids', 'attention_mask' (BERT/Bart)
                       ou tensor se "one-hot-encoder".
        """

        # === [A] Extrator de Imagem ===
        if self.cnn_model_name == ("vit-base-patch16-224"):
            inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
            outputs = self.image_encoder(**inputs)
            # outputs.last_hidden_state => (batch, seq_len_img, hidden_dim)
            image_features = outputs.last_hidden_state
        else:
            # CNN -> (batch, cnn_dim_output)
            image_features = self.image_encoder(image)
            # Dá forma (batch, 1, cnn_dim_output)
            image_features = image_features.unsqueeze(1)

        # Projeção p/ espaço comum
        b_i, s_i, d_i = image_features.shape
        image_features = image_features.view(b_i*s_i, d_i)
        projected_image_features = self.image_projector(image_features)
        image_features = projected_image_features.view(b_i, s_i, -1)
        # -> (seq_len_img, batch, common_dim)
        image_features = image_features.permute(1, 0, 2)

        # === [B] Extrator de Texto ===
        # if self.text_model_name == "one-hot-encoder":
        text_features = self.text_fc(text_metadata)  # (batch, 512)
        text_features = text_features.unsqueeze(1) # Adiciona uma dimensão às features
        # Projeção para espaço comum
        b_tt, s_tt, d_tt = text_features.shape
        text_features = text_features.view(b_tt*s_tt, d_tt)
        projected_text_features = self.text_projector(text_features)
        text_features = projected_text_features.view(b_tt, s_tt, -1)
        text_features = text_features.permute(1, 0, 2)

        # === [C] Self-Attention Intra-Modality ===
        image_features_att, _ = self.image_self_attention(
            image_features, image_features, image_features
        )
        text_features_att, _ = self.text_self_attention(
            text_features, text_features, text_features
        )

        # === [D] Cross-Attention Inter-Modality ===
        # "Imagem assiste ao texto"
        image_cross_att, _ = self.image_cross_attention(
            query=image_features_att,
            key=text_features_att,
            value=text_features_att
        )
        # "Texto assiste à imagem"
        text_cross_att, _ = self.text_cross_attention(
            query=text_features_att,
            key=image_features_att,
            value=image_features_att
        )

        # === [E] Pooling das atenções finais 
        image_cross_att = image_cross_att.permute(1, 0, 2)  # (batch, seq_len_img, common_dim)
        text_cross_att = text_cross_att.permute(1, 0, 2)    # (batch, seq_len_text, common_dim)

        image_pooled = image_cross_att.mean(dim=1)  # (batch, common_dim)
        text_pooled = text_cross_att.mean(dim=1)    # (batch, common_dim)

        combined_features = torch.cat([image_pooled, text_pooled], dim=1)

        output = self.fc_fusion(combined_features)  # (batch, num_classes)
        return output
