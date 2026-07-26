import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
    vocab_size=91,
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
    df = df.fillna("EMPTY")
    df = df.replace(r"^\s*$", "EMPTY", regex=True)
    df = df.replace(" ", "EMPTY").replace("  ", "EMPTY")
    df = df.replace("NÃO  ENCONTRADO", "EMPTY")
    df = df.replace("BRASIL", "BRAZIL")
    return df

def parse_csv_line_to_cols(text_line: str, columns: list) -> pd.DataFrame:
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

    with open(ohe_path, "rb") as f:
        ohe = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    categorical_data = ohe.transform(features[categorical_cols])
    numerical_data = scaler.transform(features[NUMERICAL_COLS])
    processed = np.hstack([categorical_data, numerical_data])

    return torch.tensor(processed, dtype=torch.float32).to(device)


# ==========================================================
# METRICS: Entropy, KL, JS
# ==========================================================
def safe_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return p

def entropy(p: np.ndarray, base: float = math.e) -> float:
    p = safe_probs(p)
    h = -np.sum(p * np.log(p))
    if base != math.e:
        h = h / np.log(base)
    return float(h)

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # KL(p || q)
    p = safe_probs(p)
    q = safe_probs(q)
    return float(np.sum(p * np.log(p / q)))

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = safe_probs(p)
    q = safe_probs(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ==========================================================
# PLOTTING (no seaborn, no fixed colors)
# ==========================================================
def plot_line(x_labels, y1, y2, label1, label2, title, ylabel, out_path, mark_idx_1=None, mark_idx_2=None):
    plt.figure(figsize=(10, 5))
    plt.plot(x_labels, y1, marker="o", label=label1)
    plt.plot(x_labels, y2, marker="s", label=label2)

    if mark_idx_1:
        for i in mark_idx_1:
            plt.scatter(x_labels[i], y1[i], marker="x")
    if mark_idx_2:
        for i in mark_idx_2:
            plt.scatter(x_labels[i], y2[i], marker="x")

    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# ==========================================================
# MAIN ANALYSIS
# ==========================================================
def run_analysis_for_weight_status(
    device,
    model,
    image_tensor,
    encoder_dir,
    class_list,
    text_configurations: dict,
    baseline_key: str = "original_metadata"
) -> pd.DataFrame:
    rows = []

    # baseline
    base_meta = process_metadata_pad20(text_configurations[baseline_key], encoder_dir, device)
    with torch.no_grad():
        base_logits = model(image_tensor, base_meta)
        base_probs_t = torch.softmax(base_logits, dim=1)[0]
    base_probs = base_probs_t.detach().cpu().numpy()
    base_pred = int(np.argmax(base_probs))
    base_conf = float(base_probs[base_pred])

    for name, meta_text in text_configurations.items():
        meta = process_metadata_pad20(meta_text, encoder_dir, device)

        with torch.no_grad():
            logits = model(image_tensor, meta)
            probs_t = torch.softmax(logits, dim=1)[0]

        probs = probs_t.detach().cpu().numpy()
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        # metrics
        ent = entropy(probs)  # nats
        kl = kl_divergence(probs, base_probs)  # KL(current || baseline)
        js = js_divergence(probs, base_probs)

        # class-change related
        class_flip = int(pred != base_pred)
        delta_conf_baseclass = float(probs[base_pred] - base_conf)

        row = {
            "configuration": name,
            "pred_class": pred,
            "pred_label": class_list[pred],
            "confidence": conf,
            "entropy": ent,
            "kl_divergence": kl,
            "js_divergence": js,
            "baseline_pred_class": base_pred,
            "baseline_pred_label": class_list[base_pred],
            "baseline_confidence": base_conf,
            "class_flip": class_flip,
            "delta_conf_baseline_class": delta_conf_baseclass,
        }

        # store probs per class too
        for k, cls in enumerate(class_list):
            row[f"prob_{cls}"] = float(probs[k])
            row[f"baseprob_{cls}"] = float(base_probs[k])

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- paths/config ----
    image_path = "./data/PAD-UFES-20/images/PAT_46_881_14.png"
    encoder_dir = "./data/preprocess_data"
    class_list = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]

    out_dir = "./results/XAI"
    os.makedirs(out_dir, exist_ok=True)

    # ---- metadata configs ----
    text_configurations = {
        "original_metadata": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,True,True,True,PAT_46_881_14.png,True",
        "age": "PAT_46,881,False,False,POMERANIA,POMERANIA,80,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,True,True,True,PAT_46_881_14.png,True",
        "grew": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,False,False,True,True,True,PAT_46_881_14.png,True",
        "bleed": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,True,False,True,PAT_46_881_14.png,True",
        "changed": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,False,True,True,PAT_46_881_14.png,True",
        "elevation": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,False,True,True,False,PAT_46_881_14.png,True",
        "itch": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,False,True,False,True,True,True,PAT_46_881_14.png,True",
        "hurt": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,NECK,6.0,5.0,BCC,True,True,True,True,True,True,PAT_46_881_14.png,True",
        "region": "PAT_46,881,False,False,POMERANIA,POMERANIA,55,False,FEMALE,True,True,True,True,3.0,FACE,6.0,5.0,BCC,True,True,False,True,True,True,PAT_46_881_14.png,True",
    }

    # ---- image ----
    img_pil, image_tensor = process_image(image_path, device)

    # ---- run for both training regimes ----
    all_dfs = {}

    for weight_status in ["frozen_weights", "unfrozen_weights"]:
        model_path = (
            "./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/"
            f"{weight_status}/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/"
            "densenet169_fold_2/model.pth"
        )

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

        df = run_analysis_for_weight_status(
            device=device,
            model=model,
            image_tensor=image_tensor,
            encoder_dir=encoder_dir,
            class_list=class_list,
            text_configurations=text_configurations,
            baseline_key="original_metadata",
        )

        csv_path = os.path.join(out_dir, f"prediction_analysis_{weight_status}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

        all_dfs[weight_status] = df

    # ---- plotting (compare frozen vs unfrozen) ----
    df_frozen = all_dfs["frozen_weights"].copy()
    df_unfrozen = all_dfs["unfrozen_weights"].copy()

    # ensure same order
    configs = df_frozen["configuration"].tolist()

    ent_f = df_frozen["entropy"].values
    ent_u = df_unfrozen["entropy"].values

    kl_f = df_frozen["kl_divergence"].values
    kl_u = df_unfrozen["kl_divergence"].values

    js_f = df_frozen["js_divergence"].values
    js_u = df_unfrozen["js_divergence"].values

    flips_f = df_frozen["class_flip"].values
    flips_u = df_unfrozen["class_flip"].values

    idx_flip_f = [i for i, v in enumerate(flips_f) if int(v) == 1]
    idx_flip_u = [i for i, v in enumerate(flips_u) if int(v) == 1]

    # Entropy plot
    plot_line(
        x_labels=configs,
        y1=ent_f,
        y2=ent_u,
        label1="Frozen",
        label2="Unfrozen",
        title="Entropy Variation Under Metadata Changes",
        ylabel="Prediction Entropy (nats)",
        out_path=os.path.join(out_dir, "entropy_variation_plot.png"),
        mark_idx_1=idx_flip_f,
        mark_idx_2=idx_flip_u
    )

    # KL plot
    plot_line(
        x_labels=configs,
        y1=kl_f,
        y2=kl_u,
        label1="Frozen",
        label2="Unfrozen",
        title="KL Divergence vs Baseline (original_metadata)",
        ylabel="KL(current || baseline)",
        out_path=os.path.join(out_dir, "kl_divergence_plot.png"),
        mark_idx_1=idx_flip_f,
        mark_idx_2=idx_flip_u
    )

    # JS plot
    plot_line(
        x_labels=configs,
        y1=js_f,
        y2=js_u,
        label1="Frozen",
        label2="Unfrozen",
        title="Jensen-Shannon Divergence vs Baseline (original_metadata)",
        ylabel="JS divergence",
        out_path=os.path.join(out_dir, "js_divergence_plot.png"),
        mark_idx_1=idx_flip_f,
        mark_idx_2=idx_flip_u
    )

    # Optional: Delta confidence on baseline class
    dc_f = df_frozen["delta_conf_baseline_class"].values
    dc_u = df_unfrozen["delta_conf_baseline_class"].values

    plot_line(
        x_labels=configs,
        y1=dc_f,
        y2=dc_u,
        label1="Frozen",
        label2="Unfrozen",
        title="Change in Baseline-Class Probability Under Metadata Changes",
        ylabel="Δ P(baseline_class)",
        out_path=os.path.join(out_dir, "delta_baseline_prob_plot.png"),
        mark_idx_1=idx_flip_f,
        mark_idx_2=idx_flip_u
    )