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
from benchmark.models import multimodalIntraInterModal

# ===== Utility Functions =====

def load_transforms(image_encoder="densenet169"):
    if image_encoder == "vit-base-patch16-224":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
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

def process_data(text, column_names):
    data = pd.DataFrame([text.split(',')], columns=column_names)
    data = data.fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").replace("NÃO  ENCONTRADO", "EMPTY")
    data = data.replace(r'^\s*$', 'EMPTY', regex=True)
    data = data.replace(['NÃO ENCONTRADO', 'n/a', None], 'EMPTY')
    data = data.replace("BRASIL", "BRAZIL")
    return data

def one_hot_encoding(metadata):
    dataset_features = metadata.drop(columns=['patient_id', 'lesion_id', 'img_id', 'biopsed', 'diagnostic'])
    for col in ['age', 'diameter_1', 'diameter_2', 'fitspatrick']:
        dataset_features[col] = pd.to_numeric(dataset_features[col], errors='coerce')
    categorical_cols = dataset_features.select_dtypes(include=['object', 'bool']).columns
    numerical_cols = dataset_features.select_dtypes(include=['float64', 'int64']).columns
    dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)
    dataset_features[numerical_cols] = dataset_features[numerical_cols].fillna(-1)
    print(f"{dataset_features[numerical_cols]}\n")
    dataset_features_categorical = dataset_features[categorical_cols]
    ohe_path = "./src/results/preprocess_data/ohe.pickle"
    if os.path.exists(ohe_path):
        with open(ohe_path, 'rb') as f:
            ohe = pickle.load(f)
        categorical_data = ohe.transform(dataset_features_categorical)
    else:
        raise FileNotFoundError("OneHotEncoder file not found.")
    scaler_path = "./src/results/preprocess_data/scaler.pickle"
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        numerical_data = scaler.transform(dataset_features[numerical_cols])
    else:
        raise FileNotFoundError("StandardScaler file not found.")
    processed_metadata = np.hstack((categorical_data, numerical_data))
    return processed_metadata

def resize_heatmap(heatmap, target_size):
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(heatmap_tensor, size=target_size, mode='bilinear', align_corners=False)
    return heatmap_resized.squeeze().numpy()

# ===== GradCAM Implementation Using torch.autograd.grad =====

class GradCAM:
    def __init__(self, model, target_layer, device):
        """
        model: The loaded multimodal model.
        target_layer: The layer whose activations will be used for GradCAM.
        device: The device ('cuda' ou 'cpu').
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device

    def forward_with_activation(self, image, metadata):
        """
        Executa uma passagem forward capturando as ativações da camada alvo.
        Retorna:
            output: Saída do modelo.
            activations: Ativações capturadas da camada alvo.
        """
        activations = None

        def hook(module, input, output):
            nonlocal activations
            activations = output  # Permite que os gradientes fluam (sem detach).

        handle = self.target_layer.register_forward_hook(hook)
        output = self.model(image, metadata)
        handle.remove()

        if activations is None:
            raise RuntimeError("Não foi possível capturar as ativações. Verifique a camada alvo.")
        return output, activations

    def generate_heatmap(self, image, metadata, target_class):
        """
        Gera o heatmap GradCAM para a classe alvo especificada.
        image: Tensor da imagem pré-processada (1, C, H, W).
        metadata: Tensor dos metadados processados.
        target_class: Índice da classe alvo.
        Retorna:
            Heatmap como um array numpy.
        """
        image.requires_grad = True  # Garante que os gradientes serão calculados

        output, activations = self.forward_with_activation(image, metadata)
        target_score = output[0, target_class]
        grads = torch.autograd.grad(target_score, activations, retain_graph=True)[0]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, fH, fW)
        cam = F.relu(cam)
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min != 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam.zero_()
        _, _, H, W = image.shape
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().detach().numpy()

# ===== Seção Principal =====

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/optimize-num-heads/stratifiedkfold/frozen-weights/2/no-metadata/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_5_20250213_113702/model.pth"
    # model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20250108_170320/model.pth"
    #model_path="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/unfreeze-weights/2/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_5_20250112_181658/model.pth"
    # Carregar e pré-processar a imagem.
    image_path = "./PAD-UFES-20/images/PAT_8_15_820.png"
    image_pil = Image.open(image_path)
    processed_image = process_image(image_pil, image_encoder="densenet169")
    processed_image = processed_image.unsqueeze(0).to(device)
    
    # Definir nomes das colunas para os metadados.
    column_names = [
        "patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age",
        "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
        "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic", "itch",
        "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"
    ]
    
    text = "PAT_46,881,,,,,,,,,,,,,,,,BCC,,,,,True,,PAT_46_881_14.png,True"
    metadata = process_data(text, column_names)
    processed_metadata = one_hot_encoding(metadata)
    processed_metadata_tensor = torch.tensor(processed_metadata, dtype=torch.float32).to(device)
    
    model = load_multimodal_model(device, model_path, "gfcam")
    target_layer = model.image_encoder.features[-1]  # Ajuste conforme necessário
    
    gradcam = GradCAM(model, target_layer, device)
    target_class = -1
    heatmap = gradcam.generate_heatmap(processed_image, processed_metadata_tensor, target_class)
    heatmap_resized = resize_heatmap(heatmap, (image_pil.height, image_pil.width))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_pil)
    axes[0].set_title("Imagem Original")
    axes[0].axis('off')
    
    axes[1].imshow(image_pil)
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.4)
    axes[1].set_title("Imagem com GradCAM")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
