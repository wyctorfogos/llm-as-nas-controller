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

# ===== Utility Functions (same as before) =====

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

def load_multimodal_model(device, model_path, attention_mecanism, vocab_size=85, n=1):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=6,
        num_heads=2,
        device=device,
        cnn_model_name="densenet169",
        text_model_name="one-hot-encoder",
        vocab_size=vocab_size,
        attention_mecanism=attention_mecanism,
        n=n
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
    
def generated_heatmap_image(text, image_pil, device, model_path):
    try:
        metadata = process_data(text, column_names)
        processed_metadata = one_hot_encoding(metadata)
        processed_metadata_tensor = torch.tensor(processed_metadata, dtype=torch.float32).to(device)
        
        # Load multimodal model
        model = load_multimodal_model(device, model_path, "gfcam", n=2)
        
        # Choose target layer for GradCAM++
        target_layer = model.image_encoder.features[-1]  # Adjust as needed
        
        gradcam_pp = GradCAMPlusPlus(model, target_layer, device)
        target_class = -1  # Adjust target class index as needed
        
        # Generate GradCAM++ heatmap
        heatmap = gradcam_pp.generate_heatmap(processed_image, processed_metadata_tensor, target_class)
        heatmap_resized = resize_heatmap(heatmap, (image_pil.height, image_pil.width))
        return heatmap_resized
    except Exception as e:
        print(f"Erro ao processar o heatmap. Erro:{e}\n")
        pass
# ===== Main Script for GradCAM++ =====

if __name__ == "__main__":
    device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20250108_170320/model.pth"
    # model_path="/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_4_20250108_170320/model.pth"
    # model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/unfreeze-weights/2/gfcam/model_densenet169_with_one-hot-encoder_512/densenet169_fold_5_20250112_181658/model.pth"
    # model_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/src/results/86_features_metadata/optimize-num-heads/stratifiedkfold/frozen-weights/2/no-metadata/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_5_20250213_113702/model.pth"
    # model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/last-layer-unfrozen/2/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_1_20250211_103249/model.pth" # "last-layer-unfrozen-weights"
    # model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/unfrozen-weights/2/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_1_20250215_083801/model.pth" # "unfrozen-weights"
    model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/frozen-weights/2/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_4_20250215_075350/model.pth" # "frozen-weights"
    # Define column names for metadata processing
    column_names = [
        "patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age",
        "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
        "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic", "itch",
        "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"
    ]

    wanted_image_list = [
        {"class":"BCC", "image":"PAT_46_881_939.png", "age": "55", "fitspatrick":"3.0", "orig_metadata": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,True,True,True,PAT_46_881_939.png,True"}, 
        {"class":"ACK", "image":"PAT_236_361_180.png", "age": "55", "fitspatrick":"3.0", "orig_metadata": "PAT_236,361,False,True,POMERANIA,POMERANIA,55,True,MALE,True,True,False,False,3.0,CHEST,6.0,5.0,ACK,True,False,False,False,False,True,PAT_236_361_180.png,True"}, 
        {"class":"SCC", "image":"PAT_380_1540_959.png", "age": "60", "fitspatrick":"2.0", "orig_metadata": "PAT_380,1540,False,False,NETHERLANDS,GERMANY,60,True,MALE,False,True,True,True,2.0,NOSE,3.0,3.0,SCC,True,False,False,False,False,False,PAT_380_1540_959.png,True"},
        {"class":"SEK", "image":"PAT_107_160_609.png", "age": "82", "fitspatrick":"1.0", "orig_metadata": "PAT_107,160,False,False,POMERANIA,POMERANIA,82,False,FEMALE,False,False,False,False,1.0,CHEST,9.0,8.0,SEK,False,True,False,False,False,True,PAT_107_160_609.png,True"}, 
        {"class":"NEV", "image":"PAT_958_1812_62.png", "age": "66", "fitspatrick":"3.0", "orig_metadata": "PAT_958,1812,False,False,POMERANIA,POMERANIA,66,False,FEMALE,False,False,True,True,3.0,SCALP,17.0,15.0,NEV,True,UNK,False,UNK,False,False,PAT_958_1812_62.png,True"}, 
        {"class":"MEL", "image":"PAT_680_1289_182.png", "age": "78", "fitspatrick":"2.0", "orig_metadata": "PAT_680,1289,True,False,PORTUGAL,ITALY,78,False,MALE,True,True,True,True,2.0,BACK,10.0,10.0,MEL,False,True,False,True,False,True,PAT_680_1289_182.png,True"}
    ]
        # Create the subplots (6 images, each with 6 variations)
    fig, axes = plt.subplots(len(wanted_image_list), 13, figsize=(100, len(wanted_image_list) * 2))

    for i, item in enumerate(wanted_image_list):
        image_class = item["class"]
        image_name = item["image"]
        image_age = item["age"]
        image_fitspatrick = item["fitspatrick"]
        image_orig_metadata = item["orig_metadata"]

        # Load and preprocess image
        image_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/data/PAD-UFES-20/images/{image_name}"
        image_pil = Image.open(image_path)
        processed_image = process_image(image_pil, image_encoder="densenet169")
        processed_image = processed_image.unsqueeze(0).to(device)  # Add batch dimension

        # Generating heatmaps for various attributes
        text_orig_metadata = image_orig_metadata
        heatmap_resized_orig_metadata = generated_heatmap_image(text_orig_metadata, image_pil, device, model_path)

        # Missing data
        text_missing_metadata = f",881,,,,,,,,,,,,,,,,{image_class},,,,,,,{image_name},"
        heatmap_resized_missing_metadata = generated_heatmap_image(text_missing_metadata, image_pil, device, model_path)

        text_age = f",881,,,,,{image_age},,,,,,,,,,,{image_class},,,,,,,{image_name},"
        heatmap_resized_age = generated_heatmap_image(text_age, image_pil, device, model_path)
        
        text_grew = f",881,,,,,,,,,,,,,,,,{image_class},,True,,,,,{image_name},"
        heatmap_resized_grew = generated_heatmap_image(text_grew, image_pil, device, model_path)
        
        text_bleed = f",881,,,,,,,,,,,,,,,,{image_class},,,,,True,,{image_name},"
        heatmap_resized_bleed = generated_heatmap_image(text_bleed, image_pil, device, model_path)
        
        text_smoke = f",881,True,,,,,,,,,,,,,,,{image_class},,,,,,,{image_name},"
        heatmap_resized_smoke = generated_heatmap_image(text_smoke, image_pil, device, model_path)
        
        text_itch = f",881,,,,,,,,,,,,,,,,{image_class},True,,,,,,{image_name},"
        heatmap_resized_itch = generated_heatmap_image(text_itch, image_pil, device, model_path)

        text_elevation = f",881,,,,,,,,,,,,,,,,{image_class},,,,,,True,{image_name},"
        heatmap_resized_elevation = generated_heatmap_image(text_itch, image_pil, device, model_path)

        text_changed = f",881,,,,,,,,,,,,,,,,{image_class},,,,True,,,{image_name},"
        heatmap_resized_changed = generated_heatmap_image(text_itch, image_pil, device, model_path)
        
        text_cancer_history = f",881,,,,,,,,,True,,,,,,,{image_class},,,,,,,{image_name},"
        heatmap_resized_cancer_history = generated_heatmap_image(text_cancer_history, image_pil, device, model_path)

        text_hurt = f",881,,,,,,,,,,,,,,,,{image_class},,,True,,,,{image_name},"
        heatmap_resized_hurt = generated_heatmap_image(text_hurt, image_pil, device, model_path)

        text_fitz = f",881,,,,,,,,,,,,{image_fitspatrick},,,,{image_class},,,,,,,{image_name},"
        heatmap_resized_fitz = generated_heatmap_image(text_fitz, image_pil, device, model_path)

        # Plot the original image and heatmaps for each variation
        axes[i, 0].imshow(image_pil)
        axes[i, 0].set_title(f"Original Image - {image_class}",  fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(image_pil)
        axes[i, 1].imshow(heatmap_resized_orig_metadata, cmap='jet', alpha=0.4)
        axes[i, 1].set_title("Original metadata", fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(image_pil)
        axes[i, 2].imshow(heatmap_resized_missing_metadata, cmap='jet', alpha=0.4)
        axes[i, 2].set_title("No metadata", fontsize=10)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(image_pil)
        axes[i, 3].imshow(heatmap_resized_age, cmap='jet', alpha=0.4)
        axes[i, 3].set_title("Age", fontsize=10)
        axes[i, 3].axis('off')

        axes[i, 4].imshow(image_pil)
        axes[i, 4].imshow(heatmap_resized_grew, cmap='jet', alpha=0.4)
        axes[i, 4].set_title("Grew", fontsize=10)
        axes[i, 4].axis('off')

        axes[i, 5].imshow(image_pil)
        axes[i, 5].imshow(heatmap_resized_bleed, cmap='jet', alpha=0.4)
        axes[i, 5].set_title("Bleed", fontsize=10)
        axes[i, 5].axis('off')

        axes[i, 6].imshow(image_pil)
        axes[i, 6].imshow(heatmap_resized_smoke, cmap='jet', alpha=0.4)
        axes[i, 6].set_title("Smoke", fontsize=10)
        axes[i, 6].axis('off')

        axes[i, 7].imshow(image_pil)
        axes[i, 7].imshow(heatmap_resized_itch, cmap='jet', alpha=0.4)
        axes[i, 7].set_title("Itch", fontsize=10)
        axes[i, 7].axis('off')


        axes[i, 8].imshow(image_pil)
        axes[i, 8].imshow(heatmap_resized_elevation, cmap='jet', alpha=0.4)
        axes[i, 8].set_title("Elevation", fontsize=10)
        axes[i, 8].axis('off')

        axes[i, 9].imshow(image_pil)
        axes[i, 9].imshow(heatmap_resized_cancer_history, cmap='jet', alpha=0.4)
        axes[i, 9].set_title("Cancer history", fontsize=10)
        axes[i, 9].axis('off')

        axes[i, 10].imshow(image_pil)
        axes[i, 10].imshow(heatmap_resized_changed, cmap='jet', alpha=0.4)
        axes[i, 10].set_title("Changed", fontsize=10)
        axes[i, 10].axis('off')

        axes[i, 11].imshow(image_pil)
        axes[i, 11].imshow(heatmap_resized_hurt, cmap='jet', alpha=0.4)
        axes[i, 11].set_title("Hurt", fontsize=10)
        axes[i, 11].axis('off')

        axes[i, 12].imshow(image_pil)
        axes[i, 12].imshow(heatmap_resized_fitz, cmap='jet', alpha=0.4)
        axes[i, 12].set_title("Fitspatrick", fontsize=10)
        axes[i, 12].axis('off')

    plt.tight_layout()
    plt.show()