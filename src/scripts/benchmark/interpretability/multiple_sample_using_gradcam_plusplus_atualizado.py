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
# LOAD MODEL
# ==========================================================
def _strip_module_prefix(state_dict):
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
    vocab_size=85,
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

    # safer default for PyTorch >=2.2
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    # state_dict inside dict
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        state_dict = _strip_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
    else:
        model = ckpt

    model.to(device)
    model.eval()
    return model


# ==========================================================
# METADATA PROCESSING
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


def clean_metadata(df):
    df = df.fillna("EMPTY")
    df = df.replace(r"^\s*$", "EMPTY", regex=True)
    df = df.replace("NÃO  ENCONTRADO", "EMPTY")
    df = df.replace("BRASIL", "BRAZIL")
    return df


def parse_csv_line_to_cols(text_line):
    parts = text_line.split(",")
    if len(parts) < len(PAD_COLUMNS):
        parts += [""] * (len(PAD_COLUMNS) - len(parts))
    elif len(parts) > len(PAD_COLUMNS):
        parts = parts[:len(PAD_COLUMNS)]
    return pd.DataFrame([parts], columns=PAD_COLUMNS)


def process_metadata_pad20(text_line, encoder_dir, device, ohe, scaler):

    df = parse_csv_line_to_cols(text_line)
    df = clean_metadata(df)

    features = df.drop(columns=DROP_COLS)

    categorical_cols = [c for c in features.columns if c not in NUMERICAL_COLS]
    features[categorical_cols] = features[categorical_cols].astype(str)

    features[NUMERICAL_COLS] = (
        features[NUMERICAL_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )

    categorical_data = ohe.transform(features[categorical_cols])
    numerical_data = scaler.transform(features[NUMERICAL_COLS])

    processed = np.hstack([categorical_data, numerical_data])

    return torch.tensor(processed, dtype=torch.float32).to(device)


# ==========================================================
# FIND LAST CONV (more robust than features[-1])
# ==========================================================
def find_last_conv(module):
    last_conv = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in the image encoder.")
    return last_conv


# ==========================================================
# CAM CLASSES
# ==========================================================
class BaseCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._fh = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _normalize(self, cam):
        cam = F.relu(cam)
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def clear(self):
        self.activations = None


class GradCAMPlusPlus(BaseCAM):
    def generate(self, image, metadata, target_class):
        self.clear()

        # ensure a fresh leaf tensor with grad enabled
        image = image.clone().detach().requires_grad_(True)

        output = self.model(image, metadata)
        score = output[:, target_class]

        if self.activations is None:
            raise RuntimeError("Activations not captured. Check forward hook / target_layer.")

        # no need for higher-order graphs to produce the heatmap
        grads = torch.autograd.grad(
            score,
            self.activations,
            retain_graph=False,
            create_graph=False
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
# VARIATIONS (13 columns total)
# col 0 = original, col 1..12 = overlays
# ==========================================================
def build_variations(item):
    """
    Returns an ordered list of (title, csv_line) with 12 variations.
    All lines must have exactly len(PAD_COLUMNS) fields after parsing.
    """
    image_class = item["class"]
    image_name = item["image"]
    image_age = item["age"]
    image_region = item["region"]
    image_gender = item["gender"]
    image_fitspatrick = item["fitspatrick"]
    image_orig_metadata = item["orig_metadata"]

    # NOTE: keep lesion_id from the orig_metadata (2nd field) to avoid mismatch
    # We'll extract it safely:
    parts = image_orig_metadata.split(",")
    lesion_id = parts[1] if len(parts) > 1 and parts[1] != "" else "0"
    patient_id = parts[0] if len(parts) > 0 and parts[0] != "" else "PAT"

    def line_with(**kwargs):
        # start with all EMPTY
        row = [""] * len(PAD_COLUMNS)
        # always fill minimal identifiers + label + image
        # patient_id, lesion_id, diagnostic, img_id
        row[PAD_COLUMNS.index("patient_id")] = patient_id
        row[PAD_COLUMNS.index("lesion_id")] = lesion_id
        row[PAD_COLUMNS.index("diagnostic")] = image_class
        row[PAD_COLUMNS.index("img_id")] = image_name

        # apply overrides
        for k, v in kwargs.items():
            row[PAD_COLUMNS.index(k)] = v

        return ",".join(row)

    variations = [
        ("OrigMetadata", image_orig_metadata),
        ("NoMeta",   line_with()),
        ("Age",      line_with(age=image_age)),
        ("Grew",     line_with(grew="True")),
        ("Bleed",    line_with(bleed="True")),
        ("Smoke",    line_with(smoke="True")),
        ("Itch",     line_with(itch="True")),
        ("Elevation",line_with(elevation="True")),
        ("CancerHist",line_with(cancer_history="True")),
        ("Changed",  line_with(changed="True")),
        ("Hurt",     line_with(hurt="True")),
        ("Drink", line_with(drink="True")),
        ("Gender", line_with(gender=image_gender)),
        ("Region", line_with(region=image_region)),
        ("Fitz",     line_with(fitspatrick=image_fitspatrick)),
    ]
    return variations


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_dir = "./data/preprocess_data"
    class_list = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
    out_dir = "./results/XAI"
    os.makedirs(out_dir, exist_ok=True)

    cam_type = "gradcam++"

    # load encoders once
    ohe_path = os.path.join(encoder_dir, "ohe_pad_20.pickle")
    scaler_path = os.path.join(encoder_dir, "scaler_pad_20.pickle")
    with open(ohe_path, "rb") as f:
        ohe = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    wanted_image_list = [
        {"class":"BCC","image":"PAT_46_881_939.png", "smoke":"False","drink": "False",  "age":"55","fitspatrick":"3.0","region":"NECK", "gender":"FEMALE", 
         "orig_metadata":"PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,True,True,True,PAT_46_881_939.png,True"},
        {"class":"ACK","image":"PAT_236_361_180.png","smoke":"False","drink": "True","age":"55","fitspatrick":"3.0","region":"CHEST", "gender":"MALE",
         "orig_metadata":"PAT_236,361,False,True,POMERANIA,POMERANIA,55,True,MALE,True,True,False,False,3.0,CHEST,6.0,5.0,ACK,True,False,False,False,False,True,PAT_236_361_180.png,True"},
        {"class":"SCC","image":"PAT_380_1540_959.png","smoke":"False","drink": "False","age":"60","fitspatrick":"2.0","region":"NOSE", "gender":"MALE",
         "orig_metadata":"PAT_380,1540,False,False,NETHERLANDS,GERMANY,60,True,MALE,False,True,True,True,2.0,NOSE,3.0,3.0,SCC,True,False,False,False,False,False,PAT_380_1540_959.png,True"},
        {"class":"SEK","image":"PAT_107_160_609.png","smoke":"False","drink": "False","age":"82","fitspatrick":"1.0","region":"CHEST", "gender":"FEMALE",
         "orig_metadata":"PAT_107,160,False,False,POMERANIA,POMERANIA,82,False,FEMALE,False,False,False,False,1.0,CHEST,9.0,8.0,SEK,False,True,False,False,False,True,PAT_107_160_609.png,True"},
        {"class":"NEV","image":"PAT_958_1812_62.png","smoke":"False","drink": "False","age":"66","fitspatrick":"3.0","region":"SCALP", "gender":"FEMALE",
         "orig_metadata":"PAT_958,1812,False,False,POMERANIA,POMERANIA,66,False,FEMALE,False,False,True,True,3.0,SCALP,17.0,15.0,NEV,True,UNK,False,UNK,False,False,PAT_958_1812_62.png,True"},
        {"class":"MEL","image":"PAT_680_1289_182.png","smoke":"True","drink": "False","age":"78","fitspatrick":"2.0","region":"BACK", "gender":"ALE",
         "orig_metadata":"PAT_680,1289,True,False,PORTUGAL,ITALY,78,False,MALE,True,True,True,True,2.0,BACK,10.0,10.0,MEL,False,True,False,True,False,True,PAT_680_1289_182.png,True"}
    ]

    for weight_status in ["frozen_weights", "unfrozen_weights"]:

        model_path = f"./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/{weight_status}/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_2/model.pth"

        model = load_multimodal_model(
            device=device,
            model_path=model_path,
            vocab_size=91,  # keep consistent with your trained setup
            unfreeze_weights=weight_status
        )

        target_layer = find_last_conv(model.image_encoder)
        cam_generator = GradCAMPlusPlus(model, target_layer)

        # 13 columns: 0 original + 12 overlays
        nrows = len(wanted_image_list)
        variations_example = build_variations(wanted_image_list[0])
        ncols = 1 + len(variations_example)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 3.0*nrows))

        # normalize axes shape
        if nrows == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, item in enumerate(wanted_image_list):

            image_path = f"./data/PAD-UFES-20/images/{item['image']}"
            img_pil, image_tensor = process_image(image_path, device)

            # col 0: original
            axes[i, 0].imshow(img_pil)
            axes[i, 0].set_title(f"{item['class']} | Original", fontsize=20)
            axes[i, 0].axis("off")

            variations = build_variations(item)

            # fill columns 1..12
            for j, (vname, vline) in enumerate(variations, start=1):
                # metadata
                metadata_tensor = process_metadata_pad20(vline, encoder_dir, device, ohe, scaler)

                # choose explained class: predicted under this metadata configuration
                with torch.no_grad():
                    logits = model(image_tensor, metadata_tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    conf = probs[0, pred_class].item()

                heatmap = cam_generator.generate(image_tensor, metadata_tensor, pred_class)

                axes[i, j].imshow(img_pil)
                axes[i, j].imshow(heatmap, cmap="jet", alpha=0.4)
                axes[i, j].set_title(f"{vname}\n{class_list[pred_class]} {conf:.4f}", fontsize=20)
                axes[i, j].axis("off")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"mosaic_multiclass_{cam_type}_{weight_status}.png")
        plt.savefig(out_path, dpi=400)
        plt.close()

        print(f"Saved mosaic: {out_path}")
