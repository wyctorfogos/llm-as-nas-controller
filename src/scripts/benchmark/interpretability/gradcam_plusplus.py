import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Backend para salvar sem GUI
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import multimodalIntraInterModal

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

def load_multimodal_model(device, model_path, attention_mecanism="concatenation", cnn_model_name="resnet50", text_model_name="one-hot-encoder", num_heads=2, vocab_size=85, n=2, num_classes=6):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=num_classes, 
        device=device, 
        cnn_model_name=cnn_model_name, 
        text_model_name=text_model_name, 
        vocab_size=vocab_size,
        num_heads=num_heads,
        attention_mecanism=attention_mecanism,
        n=n
    )
    model.to(device)
    model.eval()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
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

def one_hot_encoding_isic2019(metadata):
    # Seleção das features
    dataset_features = metadata.drop(columns=['image', 'lesion_id', 'category'])
    # Convert specific columns to numeric if possible
    # Definir as colunas categóricas e numéricas corretamente
    for col in ['age_approx']:
        dataset_features[col] = pd.to_numeric(dataset_features[col], errors='coerce')

    # Identify categorical and numerical columns
    categorical_cols = dataset_features.select_dtypes(include=['object', 'bool']).columns
    numerical_cols = dataset_features.select_dtypes(include=['float64', 'int64']).columns
    # Converter categóricas para string
    dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)

    # Preencher valores faltantes nas colunas numéricas com a média da coluna
    dataset_features[numerical_cols] = dataset_features[numerical_cols].fillna(-1)

    os.makedirs('./src/results/preprocess_data', exist_ok=True)

    # OneHotEncoder
    if os.path.exists("./src/results/preprocess_data/ohe_isic2019.pickle"):
        with open('./src/results/preprocess_data/ohe_isic2019.pickle', 'rb') as f:
            ohe = pickle.load(f)
        categorical_data = ohe.transform(dataset_features[categorical_cols])
    else:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_data = ohe.fit_transform(dataset_features[categorical_cols])
        with open('./src/results/preprocess_data/ohe_isic2019.pickle', 'wb') as f:
            pickle.dump(ohe, f)

    # StandardScaler
    if os.path.exists("./src/results/preprocess_data/scaler_isic2019.pickle"):
        with open('./src/results/preprocess_data/scaler_isic2019.pickle', 'rb') as f:
            scaler = pickle.load(f)
        numerical_data = scaler.transform(dataset_features[numerical_cols])
    else:
        scaler = StandardScaler()
        numerical_data = scaler.fit_transform(dataset_features[numerical_cols])
        with open('./src/results/preprocess_data/scaler_isic2019.pickle', 'wb') as f:
            pickle.dump(scaler, f)

    # Concatenar dados
    processed_data = np.hstack((categorical_data, numerical_data))

    # Labels
    labels = metadata['category'].values
    if os.path.exists("./src/results/preprocess_data/label_encoder_isic2019.pickle"):
        with open('./src/results/preprocess_data/label_encoder_isic2019.pickle', 'rb') as f:
            label_encoder = pickle.load(f)
        encoded_labels = label_encoder.transform(labels)
    else:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        with open('./src/results/preprocess_data/label_encoder_isic2019.pickle', 'wb') as f:
            pickle.dump(label_encoder, f)

    return processed_data, encoded_labels, metadata['category'].unique()


def resize_heatmap(heatmap, target_size):
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(heatmap_tensor, size=target_size, mode='bilinear', align_corners=False)
    return heatmap_resized.squeeze().numpy()

# ===== GradCAM++ Implementation =====

class GradCAMPlusPlus:
    def __init__(self, model, target_layer, device):
        """
        model: The loaded multimodal model.
        target_layer: The convolutional layer whose activations will be used.
        device: 'cuda' or 'cpu'
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device

    def forward_with_activation(self, image, metadata):
        """
        Performs a forward pass while capturing the target layer's activations.
        Returns:
            output: The model's output.
            activations: The raw activations from the target layer.
        """
        activations = None

        def hook(module, input, output):
            nonlocal activations
            activations = output  # Do not detach to allow gradient flow

        handle = self.target_layer.register_forward_hook(hook)
        output = self.model(image, metadata)
        handle.remove()

        if activations is None:
            raise RuntimeError("Could not capture activations. Check the target layer.")
        return output, activations

    def generate_heatmap(self, image, metadata, target_class):
        """
        Generates a GradCAM++ heatmap for the specified target class.
        image: Preprocessed image tensor (1, C, H, W).
        metadata: Processed metadata tensor.
        target_class: The index of the target class.
        Returns:
            Heatmap as a numpy array.
        """
        image.requires_grad = True  # Ensure gradients flow

        # Forward pass capturing activations
        output, activations = self.forward_with_activation(image, metadata)
        target_score = output[0, target_class]

        # Compute the first-order gradients w.r.t. the activations
        grads = torch.autograd.grad(target_score, activations, retain_graph=True, create_graph=True)[0]
        # Compute second and third order derivatives element-wise
        grads2 = grads ** 2
        grads3 = grads ** 3

        # Compute the alpha coefficients:
        # For each pixel (i,j) in each channel k:
        #   alpha_k^{ij} = grads2 / (2 * grads2 + activations * grads3)
        # We sum the term activations*grads3 over spatial dimensions (i,j)
        denominator = 2 * grads2 + torch.sum(activations * grads3, dim=(2, 3), keepdim=True) + 1e-7
        alpha = grads2 / denominator

        # Weights are computed as the sum over spatial dimensions of (alpha * ReLU(grads))
        relu_grads = F.relu(grads)
        weights = torch.sum(alpha * relu_grads, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)

        # Weighted combination of activations (over channels)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # Shape: (1, 1, fH, fW)
        cam = F.relu(cam)

        # Normalize the heatmap to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Upsample the heatmap to the input image size
        _, _, H, W = image.shape
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().detach().numpy()

def get_colmns_format(dataset_name: str = "PAD-UFES-20"):
    dataset_name_dict=[
        { "dataset_name": "PAD-UFES-20", "columns": ["patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age",
        "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
        "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic", "itch",
        "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"] },
        { "dataset_name": "ISIC-2019", "columns": ["image", "category", "age_approx", "anatom_site_general","lesion_id", "sex"] }
    ]
    # Verifica se há um dicionário válido em relação aos dados
    for i, elem in enumerate(dataset_name_dict):
        if elem["dataset_name"]==dataset_name:
            aux = elem["columns"]
    return aux
# ===== Main Script for GradCAM++ =====

if __name__ == "__main__":
    device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    ## Escolha o modelo a ser usado
    # model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20250108_170320/model.pth"
    # model_path="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20250108_170320/model.pth"
    # model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/unfreeze-weights/2/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_5_20250112_181658/model.pth"
    # model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/optimize-num-heads/stratifiedkfold/frozen-weights/2/no-metadata/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_5_20250213_113702/model.pth"
    # model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/last-layer-unfrozen/2/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_1_20250211_103249/model.pth" # "last-layer-unfrozen-weights"
    # model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/PAD-UFES-20/unfrozen_weights/8/att-intramodal+residual+cross-attention-metadados/model_resnet-50_with_one-hot-encoder_512_with_best_architecture/resnet-50_fold_3_20250521_024514/model.pth" # "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/after_finetunning/densenet169/crossattention/model_densenet169_with_one-hot-encoder_1024/densenet169_fold_1_20250105_131137/model.pth"
    model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/stratifiedkfold/2/all-weights-unfroozen/for_test/PAD-UFES-20/unfrozen_weights/2/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_1_20250526_175732/model.pth"
    # model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/unfrozen-weights/2/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_3_20250211_093309/model.pth" # "frozen-weights"
    ## ISIC-2019
    # model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/ISIC2019/stratifiedkfold/2/all-weights-unfrozen/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_4_20250206_225539/model.pth"
    ## model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/ISIC2019/stratifiedkfold/2/all-weights-unfroozen/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_4_20250205_205810/model.pth"
    # Load and preprocess image
    image_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/data/PAD-UFES-20/images/PAT_10_18_830.png" # "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/data/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0000001.jpg"
    image_pil = Image.open(image_path)
    processed_image = process_image(image_pil, image_encoder="densenet169")
    processed_image = processed_image.unsqueeze(0).to(device)  # Add batch dimension
    
    dataset_name="PAD-UFES-20"
    
    if dataset_name == "ISIC-2019":
        # ISIC-2019
        text = "ISIC_0000000,NV,69.0,anterior torso,,female"
        column_names=get_colmns_format(dataset_name=dataset_name) #"ISIC-2019")
        metadata = process_data(text, column_names)
        processed_metadata, _, _ = one_hot_encoding_isic2019(metadata) # one_hot_encoding(metadata)
        processed_metadata_tensor = torch.tensor(processed_metadata, dtype=torch.float32).to(device)
        # Load multimodal model
        model = load_multimodal_model(device, model_path, "gfcam", vocab_size=13, n=2, num_classes=8)
    else:
        # Define column names for metadata processing
        ## PAD-UFES-20
        # text = "PAT_46,881,,,,,30,,,,,,,,,,,BCC,,True,,,,,PAT_46_881_14.png,"
        text = "PAT_10,18,True,False,GERMANY,GERMANY,69,False,MALE,False,False,True,False,1.0,ARM,13.0,9.0,SCC,True,UNK,False,UNK,False,True,PAT_10_18_830.png,True"
        column_names=get_colmns_format(dataset_name=dataset_name) #"ISIC-2019")
        metadata = process_data(text, column_names)
        processed_metadata = one_hot_encoding(metadata)
        processed_metadata_tensor = torch.tensor(processed_metadata, dtype=torch.float32).to(device)
        # Load multimodal model
        model = load_multimodal_model(device=device, model_path=model_path, cnn_model_name="densenet169", attention_mecanism="att-intramodal+residual+cross-attention-metadados", vocab_size=85, n=2, num_classes=6, num_heads=8)
        
    # Choose target layer for GradCAM++
    target_layer = model.image_encoder.features[-1]  # Adjust as needed
    
    gradcam_pp = GradCAMPlusPlus(model, target_layer, device)
    target_class = -1  # Adjust target class index as needed
    
    # Generate GradCAM++ heatmap
    heatmap = gradcam_pp.generate_heatmap(processed_image, processed_metadata_tensor, target_class)
    heatmap_resized = resize_heatmap(heatmap, (image_pil.height, image_pil.width))
    
    # Visualize the results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(image_pil)
    axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.4)
    axes[1].set_title(f"Image with GradCAM++ - {dataset_name}")
    axes[1].axis('off')

    plt.tight_layout()

    # SALVANDO AO INVÉS DE MOSTRAR
    output_path = f"gradcam_{dataset_name}.png"
    plt.savefig(output_path)
    print(f"Salvo em: {output_path}")
