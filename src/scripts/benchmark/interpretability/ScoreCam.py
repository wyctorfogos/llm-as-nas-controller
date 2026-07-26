import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../benchmark')))
from benchmark.models import multimodalIntraInterModal, multimodalIntraInterModalToOptimzeAfterFIneTunning

# === Utility Functions from Your Previous Code (abbreviated) ===

def load_transforms(image_encoder="densenet169"):
    # Use the same transforms as in your inference script
    if image_encoder == "vit-base-patch16-224":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(360),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return transform

def process_image(img, image_encoder="densenet169"):
    image = img.convert("RGB")
    transform = load_transforms(image_encoder)
    return transform(image)

def load_multimodal_model(device, model_path, attention_mecanism):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=6,
        num_heads=2,
        device=device,
        cnn_model_name="densenet169",
        text_model_name="one-hot-encoder",
        vocab_size=86,
        attention_mecanism=attention_mecanism
    )
    model.to(device)
    model.eval()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model

# === ScoreCAM Implementation ===

class ScoreCAM:
    def __init__(self, model, target_layer, device):
        """
        model: the loaded multimodal model.
        target_layer: the layer of the model to hook for feature maps.
        device: 'cuda' or 'cpu'.
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.features = None
        self.hook_handle = self.target_layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # Save the feature maps from the target layer
        self.features = output.detach()

    def remove_hook(self):
        self.hook_handle.remove()

    def forward(self, image, metadata):
        """
        Forward pass through the model to get the initial prediction and features.
        """
        # Reset features to None before forward
        self.features = None
        output = self.model(image, metadata)
        return output

    def generate_heatmap(self, image, metadata, target_class):
        """
        Applies ScoreCAM to generate a heatmap for the given image and target class.
        image: preprocessed image tensor with shape (1, C, H, W).
        metadata: processed metadata tensor.
        target_class: index of class for which to generate the heatmap.
        """
        # 1. Forward pass to get features and initial prediction
        _ = self.forward(image, metadata)
        feature_maps = self.features  # shape: (batch, channels, fH, fW)

        # 2. Get spatial dimensions and upsample factor
        _, num_channels, fH, fW = feature_maps.shape
        _, _, H, W = image.shape
        upsample = torch.nn.Upsample(size=(H, W), mode='bilinear')
        # 3. Initialize the score weights list and collect weighted activation maps
        score_weights = []
        weighted_maps = []

        # Iterate over each channel in the feature map
        for i in range(num_channels):
            # Extract a single channel feature map and upsample it
            fm = feature_maps[:, i:i+1, :, :]  # shape: (1, 1, fH, fW)
            cam = upsample(fm)  # shape: (1, 1, H, W)

            # Normalize the CAM to [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min != 0:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam.zero_()

            # Create a masked image by element-wise multiplication
            # Expand cam to 3 channels if necessary
            if cam.shape[1] == 1 and image.shape[1] == 3:
                cam_expanded = cam.repeat(1, 3, 1, 1)
            else:
                cam_expanded = cam

            masked_image = image * cam_expanded

            # Forward pass with the masked image
            with torch.no_grad():
                output = self.model(masked_image, metadata)
                # Apply softmax to get probabilities
                probs = torch.softmax(output, dim=1)
                target_score = probs[0, target_class].item()

            # Save weight and corresponding activation map
            score_weights.append(target_score)
            weighted_maps.append(cam.squeeze().cpu().numpy())

        # 4. Combine weighted activation maps
        score_weights = np.array(score_weights)
        # Weighted combination of maps: shape (H, W)
        combined_map = np.zeros_like(weighted_maps[0])
        for weight, w_map in zip(score_weights, weighted_maps):
            combined_map += weight * w_map

        # Apply ReLU to the combined map
        combined_map = np.maximum(combined_map, 0)

        # Normalize the final heatmap
        heatmap = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min())
        return heatmap

def process_data(text, column_names):
    # Dividir o texto em campos separados por vírgula
    data = pd.DataFrame([text.split(',')], columns=column_names)
    # Substituir valores vazios ou indesejados por "EMPTY"
    data=data.fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").\
           replace("NÃO  ENCONTRADO", "EMPTY")
    data = data.replace(r'^\s*$', 'EMPTY', regex=True)  # Substituir strings vazias ou espaços
    data = data.replace(['NÃO ENCONTRADO', 'n/a', None], 'EMPTY')  # Substituir indicadores comuns de valores vazios
    data = data.replace("BRASIL", "BRAZIL")
    return data


def one_hot_encoding(metadata):
    # Remover colunas desnecessárias e selecionar as features
    dataset_features = metadata.drop(columns=['patient_id', 'lesion_id', 'img_id', 'biopsed', 'diagnostic'])

    # Definir as colunas categóricas e numéricas corretamente
    for col in ['age', 'diameter_1', 'diameter_2', 'fitspatrick']:
        dataset_features[col] = pd.to_numeric(dataset_features[col], errors='coerce')

    # Identify categorical and numerical columns
    categorical_cols = dataset_features.select_dtypes(include=['object', 'bool']).columns
    numerical_cols = dataset_features.select_dtypes(include=['float64', 'int64']).columns
    # Converter categóricas para string
    dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)

    # Preencher valores faltantes nas colunas numéricas com a média da coluna
    dataset_features[numerical_cols] = dataset_features[numerical_cols].fillna(-1)
    print(f"{dataset_features[numerical_cols]}\n")
    # Assegurar que as colunas categóricas usadas na inferência correspondem às usadas no treinamento
    dataset_features_categorical = dataset_features[categorical_cols]

    # OneHotEncoder
    ohe_path = "./src/results/preprocess_data/ohe.pickle"
    if os.path.exists(ohe_path):
        with open(ohe_path, 'rb') as f:
            ohe = pickle.load(f)
        categorical_data = ohe.transform(dataset_features_categorical)
    else:
        raise FileNotFoundError("Arquivo OneHotEncoder não encontrado. Certifique-se de que o modelo foi treinado corretamente.")

    # StandardScaler para dados numéricos
    scaler_path = "./src/results/preprocess_data/scaler.pickle"
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        numerical_data = scaler.transform(dataset_features[numerical_cols])
    else:
        raise FileNotFoundError("Arquivo StandardScaler não encontrado. Certifique-se de que o modelo foi treinado corretamente.")

    # Concatenar os dados processados
    processed_metadata = np.hstack((categorical_data, numerical_data))
    return processed_metadata

def resize_heatmap(heatmap, target_size):
    """
    Resize heatmap to match the size of the original image.
    Args:
        heatmap (np.ndarray): Heatmap with shape (H, W).
        target_size (tuple): Target size (height, width) of the original image.
    Returns:
        np.ndarray: Resized heatmap.
    """
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    heatmap_resized = F.interpolate(heatmap_tensor, size=target_size, mode='bilinear', align_corners=False)
    return heatmap_resized.squeeze().numpy()  # Remove batch and channel dimensions


# === Main Script for Inference with ScoreCAM ===

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20250108_170320/model.pth" # "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/after_finetunning/densenet169/crossattention/model_densenet169_with_one-hot-encoder_1024/densenet169_fold_1_20250105_131137/model.pth"
    model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/unfreeze-weights/2/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_5_20250112_181658/model.pth"
    # model_path="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20250108_170320/model.pth"
    # model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/86_features_metadata/optimize-num-heads/8/gfcam/model_densenet169_with_one-hot-encoder_512_last_3_layers_unfrozen_with_best_architecture/densenet169_fold_4_20250115_122657/model.pth"
    # model_path="/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/86_features_metadata/optimize-num-heads/8/gfcam/model_densenet169_with_one-hot-encoder_512_last_layer_unfrozen_with_best_architecture/densenet169_fold_3_20250115_145328/model.pth"
    # Load and preprocess image
    image_path="./PAD-UFES-20/images/PAT_46_881_14.png"
    image_pil = Image.open(image_path)
    processed_image = process_image(image_pil, image_encoder="densenet169")
    processed_image = processed_image.unsqueeze(0).to(device)  # Add batch dimension

    # Assuming metadata preprocessing from earlier script
    # ... (Load and process metadata as earlier)
    # For simplicity, we create a dummy metadata tensor:
    # dummy_metadata = torch.zeros(1, 86).to(device)  # Adjust shape as needed for your model

    # Definir nomes das colunas
    column_names = [
        "patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age",
        "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
        "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic", "itch",
        "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"
    ]

    # Carregar dados de teste
    text="PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,True,True,True,PAT_46_881_14.png,True"
    metadata = process_data(text, column_names)

    # Processar metadados
    processed_metadata = one_hot_encoding(metadata)
    print(f"Processed_metadata:{processed_metadata}\n")

    # Load model
    model = load_multimodal_model(device, model_path, "gfcam")

    # Choose a target layer for ScoreCAM
    # For DenseNet, for example, hook the last convolutional layer:
    target_layer = model.image_encoder.features[-1]  # Adjust based on your architecture

    # Initialize ScoreCAM with the model, target layer, and device
    scorecam = ScoreCAM(model, target_layer, device)

    # Select target class index for which to generate a heatmap
    target_class = -1  # Change as needed

    # Generate heatmap using ScoreCAM
    heatmap = scorecam.generate_heatmap(processed_image, torch.tensor(processed_metadata, dtype=torch.float32).to(device), target_class)

    # Remove hook after use
    scorecam.remove_hook()


    # Converter imagem PIL para numpy
    image_np = np.array(image_pil)

    # Aplicar colormap ao heatmap
    heatmap_resized = resize_heatmap(heatmap, (image_pil.height, image_pil.width))

    # Visualizar a imagem original e a do ScoreCAM juntas
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Subplot 1: Imagem Original
    axes[0].imshow(image_pil)
    axes[0].set_title("Imagem Original")
    axes[0].axis('off')

    # Subplot 2: Imagem com ScoreCAM sobreposta
    axes[1].imshow(image_pil)  # Exibe a imagem original como fundo
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.4)  # Sobrepõe o heatmap com transparência
    axes[1].set_title("Imagem com ScoreCAM")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
