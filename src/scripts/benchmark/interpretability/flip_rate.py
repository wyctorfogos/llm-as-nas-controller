import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

from PIL import Image
from torchvision import transforms

# ==========================================================
# PATH SETUP
# ==========================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models import multimodalIntraInterModal

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "./data/PAD-UFES-20"
WEIGHTS_STATUS = "frozen_weights"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")

# Medatadaos usadados na validação
VALIDATION_METADATA_PATH="./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/unfrozen_weights/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_3/predictions_eval_fold_3.csv"

# Metadados completos
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

ENCODER_DIR = "./data/preprocess_data"
MODEL_PATH = f"./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/{WEIGHTS_STATUS}/8/gfcam/model_densenet169_with_one-hot-encoder_512_with_best_architecture/densenet169_fold_3/model.pth"

OUT_DIR = f"./results/flip_analysis_dataset_{WEIGHTS_STATUS}"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_LIST = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
K = len(CLASS_LIST)

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

def process_image(path):
    img = Image.open(path).convert("RGB")
    transform = load_image_transform()
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return tensor

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
    unfreeze_weights="unfrozen_weights"
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
# METADATA PROCESSING
# ==========================================================
# Seleciionar os dados usados apenas nos dados de validação
USED_SAMPLES_ON_EVALUATION_DATASET = list(pd.read_csv(VALIDATION_METADATA_PATH)['image_name'])
PAD_COLUMNS = list(pd.read_csv(METADATA_PATH, nrows=1).columns)

NUMERICAL_COLS = ["age", "diameter_1", "diameter_2"]
DROP_COLS = ["patient_id","lesion_id","img_id","biopsed","diagnostic"]

with open(os.path.join(ENCODER_DIR, "ohe_pad_20.pickle"), "rb") as f:
    OHE = pickle.load(f)

with open(os.path.join(ENCODER_DIR, "scaler_pad_20.pickle"), "rb") as f:
    SCALER = pickle.load(f)

def dict_to_csv_line(row):
    return ",".join([str(row[col]) for col in PAD_COLUMNS])

def process_metadata_row(row_dict):
    df = pd.DataFrame([row_dict])
    features = df.drop(columns=DROP_COLS)

    categorical_cols = [c for c in features.columns if c not in NUMERICAL_COLS]
    features[categorical_cols] = features[categorical_cols].astype(str)
    features[NUMERICAL_COLS] = features[NUMERICAL_COLS].apply(pd.to_numeric, errors="coerce").fillna(-1)

    categorical_data = OHE.transform(features[categorical_cols])
    numerical_data = SCALER.transform(features[NUMERICAL_COLS])

    processed = np.hstack([categorical_data, numerical_data])
    return torch.tensor(processed, dtype=torch.float32).to(DEVICE)

# ==========================================================
# MUTATION RULES
# ==========================================================
def mutate_metadata(row, feature):
    r = copy.deepcopy(row)

    if feature in ["itch","grew","bleed","changed","hurt","elevation",
                   "smoke","drink","skin_cancer_history","cancer_history"]:
        r[feature] = not bool(r[feature])

    elif feature in ["diameter_1","diameter_2"]:
        r[feature] = float(r[feature]) + 5
    
    elif feature == "age":
        r[feature] = 80

    elif feature == "gender":
        r[feature] = "MALE" if r[feature] == "FEMALE" else "FEMALE"

    elif feature == "region":
        r[feature] = "FACE" if r[feature] != "FACE" else "FOREARM"

    return r

# ==========================================================
# PREDICTION
# ==========================================================
def predict_class(model, image_tensor, metadata_tensor):
    with torch.no_grad():
        logits = model(image_tensor, metadata_tensor)
        probs = torch.softmax(logits, dim=1)
        return int(torch.argmax(probs, dim=1).item())

# ==========================================================
# MAIN FLIP ANALYSIS
# ==========================================================
def run_flip_analysis():

    print("Loading model...")
    model = load_multimodal_model(
                device=DEVICE,
                model_path=MODEL_PATH,
                attention_mecanism="gfcam",
                cnn_model_name="densenet169",
                vocab_size=91,
                num_heads=8,
                n=2,
                num_classes=6,
                unfreeze_weights=WEIGHTS_STATUS
            )
    print("Loading metadata...")
    # Carrega lista de imagens usadas na validação
    # Carrega metadados completos
    df_meta = pd.read_csv(METADATA_PATH)

    # Carrega lista de imagens usadas na validação
    df_val = pd.read_csv(VALIDATION_METADATA_PATH)
    used_images = df_val["image_name"].tolist()

    # Filtra apenas amostras usadas na validação
    df_meta = df_meta[df_meta["img_id"].isin(used_images)].reset_index(drop=True)

    print(f"Amostras após filtro: {len(df_meta)}")
    feature_list = [
        "age", "gender", "region", "itch","grew","bleed","changed","hurt","elevation",
        "smoke","drink","skin_cancer_history","cancer_history", "diameter_1","diameter_2"
    ]

    flip_counts = {f:0 for f in feature_list}
    transitions = {f:np.zeros((K,K), dtype=int) for f in feature_list}

    total_samples = 0

    for idx, row in df_meta.iterrows():

        image_path = os.path.join(IMAGE_DIR, row["img_id"])
        if not os.path.exists(image_path):
            continue

        image_tensor = process_image(image_path)
        metadata_dict = row.to_dict()

        baseline_tensor = process_metadata_row(metadata_dict)
        c0 = predict_class(model, image_tensor, baseline_tensor)

        for f in feature_list:
            mutated_dict = mutate_metadata(metadata_dict, f)
            mutated_tensor = process_metadata_row(mutated_dict)
            c1 = predict_class(model, image_tensor, mutated_tensor)

            transitions[f][c0, c1] += 1

            if c1 != c0:
                flip_counts[f] += 1

        total_samples += 1

        if total_samples % 100 == 0:
            print(f"Processed {total_samples} samples...")

    # ==============================
    # SUMMARY
    # ==============================
    rows = []
    for f in feature_list:
        rows.append({
            "feature": f,
            "flip_rate": flip_counts[f] / total_samples,
            "n_flips": flip_counts[f],
            "n_samples": total_samples
        })

    df_summary = pd.DataFrame(rows).sort_values("flip_rate", ascending=False)
    df_summary.to_csv(os.path.join(OUT_DIR,f"flip_summary_{WEIGHTS_STATUS}.csv"), index=False)

    # ==============================
    # BARPLOT
    # ==============================
    plt.figure(figsize=(8,6))
    plt.barh(df_summary["feature"], df_summary["flip_rate"])
    plt.xlabel("Flip Rate")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,f"flip_rate_barplot_{WEIGHTS_STATUS}.png"), dpi=400)
    plt.close()

    # ==============================
    # AGGREGATED HEATMAP
    # ==============================
    agg = np.zeros((K,K), dtype=int)
    for f in feature_list:
        agg += transitions[f]

    plt.figure(figsize=(6,5))
    plt.imshow(agg)
    plt.xticks(range(K), CLASS_LIST, rotation=45)
    plt.yticks(range(K), CLASS_LIST)
    plt.xlabel("After feature change")
    plt.ylabel("Baseline prediction")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,f"aggregated_transitions_{WEIGHTS_STATUS}.png"), dpi=400)
    plt.close()

    print("\nFlip analysis completed.")
    print(df_summary)

if __name__ == "__main__":
    run_flip_analysis()