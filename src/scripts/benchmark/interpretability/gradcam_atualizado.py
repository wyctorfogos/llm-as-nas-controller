import os
import sys
import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

# ==========================================================
# PATH SETUP
# ==========================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models import multimodalIntraInterModal


# ==========================================================
# IMAGE TRANSFORM
# ==========================================================
def load_image_transform(size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def process_image(path, device):
    img = Image.open(path).convert("RGB")
    transform = load_image_transform()
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor


# ==========================================================
# LOAD MODEL (ROBUST CHECKPOINT LOADING)
# ==========================================================
def _strip_module_prefix(state_dict):
    # Remove "module." caso tenha sido salvo com DataParallel/DistributedDataParallel
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) > 0 and keys[0].startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def load_multimodal_model(
    device,
    model_path,
    cnn_model_name="densenet169",
    attention_mecanism="gfcam",
    vocab_size=85,            # PAD-UFES-20 costuma ser 85 no seu setup
    num_heads=8,
    n=2,
    num_classes=6,
    unfreeze_weights="frozen_weights"
):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=num_classes,
        device=device,
        cnn_model_name=cnn_model_name,
        text_model_name="one-hot-encoder",
        vocab_size=vocab_size,
        num_heads=num_heads,
        attention_mecanism=attention_mecanism,
        n=n,
        unfreeze_weights=unfreeze_weights
    )

    ckpt = torch.load(model_path, map_location=device)

    # Caso: checkpoint dict
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Pode ser state_dict puro em dict
            # (mas aqui é arriscado: pode ser config junto)
            # Tentamos carregar como state_dict mesmo.
            state_dict = ckpt

        state_dict = _strip_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)

    # Caso: salvaram o modelo inteiro
    else:
        model = ckpt

    model.to(device)
    model.eval()
    return model


# ==========================================================
# METADATA PROCESSING (MATCH DATASET)
# ==========================================================
PAD_COLUMNS = [
    "patient_id", "lesion_id", "smoke", "drink",
    "background_father", "background_mother",
    "age", "pesticide", "gender",
    "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system",
    "fitspatrick", "region",
    "diameter_1", "diameter_2",
    "diagnostic",
    "itch", "grew", "hurt",
    "changed", "bleed",
    "elevation", "img_id", "biopsed"
]

NUMERICAL_COLS = ["age", "diameter_1", "diameter_2"]
DROP_COLS = ["patient_id", "lesion_id", "img_id", "biopsed", "diagnostic"]

def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliza strings vazias e variações comuns
    df = df.fillna("EMPTY")
    df = df.replace(r"^\s*$", "EMPTY", regex=True)
    df = df.replace(" ", "EMPTY").replace("  ", "EMPTY")
    df = df.replace("NÃO  ENCONTRADO", "EMPTY")
    df = df.replace("BRASIL", "BRAZIL")
    return df

def parse_csv_line_to_cols(text_line: str, columns: list) -> pd.DataFrame:
    """
    Garante que o split da linha tenha exatamente len(columns) campos:
    - se faltar: completa com ""
    - se sobrar: corta
    """
    parts = text_line.split(",")
    if len(parts) < len(columns):
        parts = parts + [""] * (len(columns) - len(parts))
    elif len(parts) > len(columns):
        parts = parts[:len(columns)]
    return pd.DataFrame([parts], columns=columns)

def process_metadata_pad20(text_line, encoder_dir, device):
    df = parse_csv_line_to_cols(text_line, PAD_COLUMNS)
    df = clean_metadata(df)

    features = df.drop(columns=DROP_COLS)

    categorical_cols = [c for c in features.columns if c not in NUMERICAL_COLS]

    features[categorical_cols] = features[categorical_cols].astype(str)

    features[NUMERICAL_COLS] = (
        features[NUMERICAL_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )

    ohe_path = os.path.join(encoder_dir, "ohe_pad_20.pickle")
    scaler_path = os.path.join(encoder_dir, "scaler_pad_20.pickle")

    if not os.path.exists(ohe_path):
        raise FileNotFoundError(f"OneHotEncoder not found: {ohe_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"StandardScaler not found: {scaler_path}")

    with open(ohe_path, "rb") as f:
        ohe = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    categorical_data = ohe.transform(features[categorical_cols])
    numerical_data = scaler.transform(features[NUMERICAL_COLS])

    processed = np.hstack([categorical_data, numerical_data])

    return torch.tensor(processed, dtype=torch.float32).to(device)


# ==========================================================
# FIND LAST CONV
# ==========================================================
def find_last_conv(module):
    last_conv = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found.")
    return last_conv


# ==========================================================
# BASE CAM CLASS
# ==========================================================
class BaseCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fh = self.target_layer.register_forward_hook(self._forward_hook)
        # full backward hook funciona melhor em versões recentes
        self._bh = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _normalize(self, cam):
        cam = F.relu(cam)
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def clear(self):
        self.activations = None
        self.gradients = None


# ==========================================================
# GradCAM
# ==========================================================
class GradCAM(BaseCAM):
    def generate(self, image, metadata, target_class):
        self.clear()

        image.requires_grad_(True)

        output = self.model(image, metadata)
        score = output[:, target_class]

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations. Check target_layer.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = self._normalize(cam)

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return cam.squeeze().detach().cpu().numpy()


# ==========================================================
# GradCAM++
# ==========================================================
class GradCAMPlusPlus(BaseCAM):
    def generate(self, image, metadata, target_class):
        self.clear()

        image.requires_grad_(True)

        output = self.model(image, metadata)
        score = output[:, target_class]

        # Garante que activations foi capturado
        if self.activations is None:
            raise RuntimeError("Activations not captured. Check forward hook / target_layer.")

        # GradCAM++ usa grads w.r.t activations (não depende do backward_hook)
        grads = torch.autograd.grad(
            score,
            self.activations,
            retain_graph=True,
            create_graph=True
        )[0]

        grads2 = grads ** 2
        grads3 = grads ** 3

        denominator = (
            2 * grads2 +
            torch.sum(self.activations * grads3, dim=(2, 3), keepdim=True)
            + 1e-8
        )

        alpha = grads2 / denominator
        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = self._normalize(cam)

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return cam.squeeze().detach().cpu().numpy()


# ==========================================================
# MOSAIC UTILS
# ==========================================================
def sanitize_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)
    return s

def save_mosaic(img_pil, results_list, out_path, cam_type, weight_status):
    """
    results_list: list of dicts:
      { "name": ..., "heatmap": np.ndarray, "pred_class": int, "confidence": float }
    Mosaic format: rows = configurations, cols = [original, overlay]
    """
    n = len(results_list)
    fig, axes = plt.subplots(n, 2, figsize=(12, max(3, 3*n)))

    if n == 1:
        axes = np.array([axes])  # normalize shape

    for i, r in enumerate(results_list):
        # original
        axes[i, 0].imshow(img_pil)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{r['name']} | Original")

        # overlay
        axes[i, 1].imshow(img_pil)
        axes[i, 1].imshow(r["heatmap"], cmap="jet", alpha=0.4)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(
            f"Prediction={class_list[pred_class]} | conf={r['confidence']:.3f}"
        )

    plt.suptitle(f"Mosaic {cam_type.upper()} | {weight_status}", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path = "./data/PAD-UFES-20/images/PAT_46_881_14.png"
    encoder_dir = "./data/preprocess_data"
    class_list = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
    out_dir = "./results/XAI/24022026/"
    os.makedirs(out_dir, exist_ok=True)

    img_pil, image_tensor = process_image(image_path, device)

    # patient_id,lesion_id,smoke,drink,background_father,background_mother,age,pesticide,gender,skin_cancer_history,cancer_history,has_piped_water,has_sewage_system,fitspatrick,region,diameter_1,diameter_2,diagnostic,itch,grew,hurt,changed,bleed,elevation,img_id,biopsed
    text_configurations = {
        "original_metadata": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,True,True,True,PAT_46_881_14.png,True",
        "only_age":          "PAT_46,881,,,,,80,,,,,,,,,,,BCC,,,,,,,PAT_46_881_14.png,",
        "only_grew":         "PAT_46,881,,,,,,,,,,,,,,,,BCC,,True,,,,,PAT_46_881_14.png,",
        "only_bleed":        "PAT_46,881,,,,,,,,,,,,,,,,BCC,,,,,True,,PAT_46_881_14.png,",
        "only_changed":      "PAT_46,881,,,,,,,,,,,,,,,,BCC,,,,True,,,PAT_46_881_14.png,",
        "only_elevation":    "PAT_46,881,,,,,,,,,,,,,,,,BCC,,,,,,True,PAT_46_881_14.png,",
        "only_itch":    "PAT_46,881,,,,,,,,,,,,,,,,BCC,True,,,,,,PAT_46_881_14.png,",
        "only_hurt":    "PAT_46,881,,,,,,,,,,,,,,,,BCC,,,True,,,,PAT_46_881_14.png,",
        "only_region":    "PAT_46,881,,,,,,,,,,,,,NECK,,,BCC,,,,,,,PAT_46_881_14.png,"
    }

    for cam_type in ["gradcam", "gradcam++"]:

        for weight_status in ["frozen_weights", "unfrozen_weights"]:

            model_path = f"./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/{weight_status}/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_2/model.pth"

            model = load_multimodal_model(
                device=device,
                model_path=model_path,
                attention_mecanism="gfcam",
                cnn_model_name="densenet169",
                vocab_size=91,
                num_heads=8,
                n=2,
                num_classes=6,
                unfreeze_weights=weight_status
            )

            target_layer = find_last_conv(model.image_encoder)

            if cam_type == "gradcam":
                cam_generator = GradCAM(model, target_layer)
            else:
                cam_generator = GradCAMPlusPlus(model, target_layer)

            # Guardar para mosaico
            mosaic_results = []

            for name, metadata_text in text_configurations.items():

                print(f"\n--- Testing configuration: {name} | {weight_status} ---")

                metadata_tensor = process_metadata_pad20(metadata_text, encoder_dir, device)

                with torch.no_grad():
                    logits = model(image_tensor, metadata_tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred_class].item()

                heatmap = cam_generator.generate(image_tensor, metadata_tensor, pred_class)

                # Salvar imagem individual
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                ax[0].imshow(img_pil)
                ax[0].set_title("Original")
                ax[0].axis("off")

                ax[1].imshow(img_pil)
                ax[1].imshow(heatmap, cmap="jet", alpha=0.4)
                ax[1].set_title(f"Prediction={class_list[pred_class]} | conf={confidence:.3f}")
                ax[1].axis("off")

                plt.tight_layout()

                fname = f"{cam_type}_pad20_{weight_status}_{sanitize_filename(name)}.png"
                out_path = os.path.join(out_dir, fname)
                plt.savefig(out_path, dpi=200)
                plt.close()

                print(f"Saved: {out_path}")

                mosaic_results.append({
                    "name": name,
                    "heatmap": heatmap,
                    "pred_class": pred_class,
                    "confidence": confidence
                })

            # Salvar mosaico por weight_status
            mosaic_name = f"mosaic_{cam_type}_pad20_{weight_status}_24022026.png"
            mosaic_path = os.path.join(out_dir, mosaic_name)

            # ordem consistente (a do dict)
            save_mosaic(
                img_pil=img_pil,
                results_list=mosaic_results,
                out_path=mosaic_path,
                cam_type=cam_type,
                weight_status=weight_status
            )

            print(f"Mosaic saved: {mosaic_path}")
