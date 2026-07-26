import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from models import multimodalIntraInterModal

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "./data/PAD-UFES-20"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")
ENCODER_DIR = "./data/preprocess_data"

BASE_RESULTS_DIR = (
    "./src/results/artigo_1_GFCAM/12022026/PAD-UFES-20/"
    "unfrozen_weights/8/gfcam/"
    "model_densenet169_with_one-hot-encoder_512_with_best_architecture"
)

OUT_DIR = "./results/feature_entropy_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_LIST = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
K = len(CLASS_LIST)

FEATURE_LIST = [
    "age", "region", "gender", "itch", "grew", "bleed", "changed", "elevation",
    "hurt", "smoke", "drink", "pesticide", "skin_cancer_history",
    "cancer_history", "has_piped_water", "has_sewage_system"
]

BENIGN = ["NEV", "ACK", "SEK"]
MALIGNANT = ["BCC", "SCC", "MEL"]

benign_idx = [CLASS_LIST.index(c) for c in BENIGN]
malig_idx = [CLASS_LIST.index(c) for c in MALIGNANT]

# ==========================================================
# ENTROPY
# ==========================================================
def entropy(p):
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


# ==========================================================
# LOAD DATA
# ==========================================================
df_all_meta = pd.read_csv(METADATA_PATH)

with open(os.path.join(ENCODER_DIR, "ohe_pad_20.pickle"), "rb") as f:
    OHE = pickle.load(f)

with open(os.path.join(ENCODER_DIR, "scaler_pad_20.pickle"), "rb") as f:
    SCALER = pickle.load(f)

OHE_FEATURE_NAMES = list(OHE.feature_names_in_)
SCALER_FEATURE_NAMES = list(SCALER.feature_names_in_)

# categories map para mutações mais seguras
OHE_CATEGORY_MAP = {}
if hasattr(OHE, "categories_"):
    OHE_CATEGORY_MAP = {
        feature: [str(x) for x in cats]
        for feature, cats in zip(OHE_FEATURE_NAMES, OHE.categories_)
    }


# ==========================================================
# IMAGE PROCESSING
# ==========================================================
def process_image(path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)


# ==========================================================
# METADATA PROCESSING
# ==========================================================
def process_metadata_batch(row_dicts):
    df = pd.DataFrame(row_dicts).copy()

    for c in OHE_FEATURE_NAMES:
        if c not in df.columns:
            df[c] = "unknown"

    cat_df = df[OHE_FEATURE_NAMES].astype(str)

    for c in SCALER_FEATURE_NAMES:
        if c not in df.columns:
            df[c] = -1

    num_df = df[SCALER_FEATURE_NAMES].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(-1)

    cat = OHE.transform(cat_df)
    if hasattr(cat, "toarray"):
        cat = cat.toarray()

    num = SCALER.transform(num_df)

    processed = np.hstack([cat, num]).astype(np.float32)

    return torch.tensor(processed, dtype=torch.float32, device=DEVICE)


def process_metadata(row_dict):
    return process_metadata_batch([row_dict])


# ==========================================================
# LOAD MODEL
# ==========================================================
def load_model(model_path):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=K,
        device=DEVICE,
        cnn_model_name="densenet169",
        text_model_name="one-hot-encoder",
        vocab_size=91,
        num_heads=8,
        attention_mecanism="gfcam",
        n=2,
        unfreeze_weights="unfrozen_weights"
    )

    ckpt = torch.load(model_path, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()

    return model


# ==========================================================
# MUTATION HELPERS
# ==========================================================
def get_alternative_category(feature, current_value, preferred=None):
    current_str = "" if pd.isna(current_value) else str(current_value)
    categories = OHE_CATEGORY_MAP.get(feature, [])

    if preferred is not None and preferred in categories and preferred != current_str:
        return preferred

    if current_str in categories and len(categories) > 1:
        for c in categories:
            if c != current_str:
                return c

    lower_val = current_str.lower()
    if lower_val in {"true", "1", "yes"}:
        return "False"
    if lower_val in {"false", "0", "no"}:
        return "True"

    if len(categories) > 0:
        for c in categories:
            if c != current_str:
                return c

    return current_str


# ==========================================================
# MUTATE FEATURE
# ==========================================================
def mutate_feature(row, feature):
    r = copy.deepcopy(row)

    if feature == "age":
        val = pd.to_numeric(r.get("age", 50), errors="coerce")
        if pd.isna(val):
            val = 50

        if val <= 80:
            new_val = val + 20
        else:
            new_val = val - 20

        r["age"] = int(np.clip(new_val, 0, 100))
        return r

    if feature == "gender":
        current = str(r.get("gender", "")).upper()
        preferred = "MALE" if current == "FEMALE" else "FEMALE"
        r["gender"] = get_alternative_category("gender", r.get("gender", ""), preferred=preferred)
        return r

    if feature == "region":
        r["region"] = get_alternative_category("region", r.get("region", ""), preferred="FACE")
        return r

    # Demais atributos categóricos/binários
    r[feature] = get_alternative_category(feature, r.get(feature, ""))
    return r


# ==========================================================
# PREDICTION
# ==========================================================
def predict_batch(model, img, meta_batch):
    batch_size = meta_batch.shape[0]
    img_batch = img.repeat(batch_size, 1, 1, 1)

    with torch.inference_mode():
        logits = model(img_batch, meta_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    preds = np.argmax(probs, axis=1)
    return preds, probs


def predict(model, img, meta):
    with torch.inference_mode():
        logits = model(img, meta)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred = int(np.argmax(probs))
    return pred, probs


# ==========================================================
# PLOTS
# ==========================================================
def plot_transition_matrix(cm, feature, out_path):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap="Blues")

    plt.xticks(range(K), CLASS_LIST, rotation=45)
    plt.yticks(range(K), CLASS_LIST)

    threshold = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(K):
        for j in range(K):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color=color, fontsize=10
            )

    plt.title(f"Prediction Transition Matrix - {feature}")
    plt.xlabel("Mutated Prediction")
    plt.ylabel("Original Prediction")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def plot_feature_sensitivity_map(flip_summary, entropy_summary, out_path, top_k_labels=8):
    df = flip_summary.merge(entropy_summary, on="feature").copy()

    # score simples para destacar as features mais influentes
    df["importance_score"] = (
        df["mean_flip_rate"].rank(pct=True) +
        df["mean_delta_entropy"].rank(pct=True)
    )

    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)

    x = df["mean_delta_entropy"].values
    y = df["mean_flip_rate"].values

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=70)

    # linhas de referência
    plt.axhline(y.mean(), linestyle="--")
    plt.axvline(x.mean(), linestyle="--")

    # rotular apenas as top-k mais importantes
    for _, row in df.head(top_k_labels).iterrows():
        plt.annotate(
            row["feature"],
            (row["mean_delta_entropy"], row["mean_flip_rate"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9
        )

    plt.xlabel("Mean Δ Entropy")
    plt.ylabel("Flip Rate")
    plt.title("Feature Sensitivity Map")
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def plot_entropy_variation(entropy_summary, out_path):
    df = entropy_summary.sort_values("mean_delta_entropy", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(11, 6))
    x = np.arange(len(df))

    plt.bar(x, df["mean_delta_entropy"])

    plt.xticks(x, df["feature"], rotation=45, ha="right")
    plt.ylabel("Mean Δ Entropy")
    plt.xlabel("Feature")
    plt.title("Prediction Entropy Change by Feature Perturbation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def plot_flip_rate_variation(flip_summary, out_path):
    df = flip_summary.sort_values("mean_flip_rate", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(11, 6))
    x = np.arange(len(df))

    plt.bar(x, df["mean_flip_rate"])

    plt.xticks(x, df["feature"], rotation=45, ha="right")
    plt.ylabel("Mean Flip Rate")
    plt.xlabel("Feature")
    plt.title("Prediction Flip Rate by Feature Perturbation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

def plot_clinical_risk(clinical_summary, out_path):
    df = clinical_summary.sort_values(
        "malignant_to_benign_mean", ascending=False
    ).reset_index(drop=True)

    x = np.arange(len(df))
    width = 0.38

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        df["benign_to_malignant_mean"],
        width=width,
        label="Benign → Malignant"
    )

    plt.bar(
        x + width / 2,
        df["malignant_to_benign_mean"],
        width=width,
        label="Malignant → Benign"
    )

    plt.xticks(x, df["feature"], rotation=45, ha="right")
    plt.ylabel("Transition Rate")
    plt.xlabel("Feature")
    plt.title("Clinical Transition Risk by Feature Perturbation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close()

# ==========================================================
# RUN FOLD
# ==========================================================
def run_fold(fold):
    print(f"Running fold {fold}")

    fold_dir = os.path.join(BASE_RESULTS_DIR, f"densenet169_fold_{fold}")
    model_path = os.path.join(fold_dir, "model.pth")
    pred_csv_path = os.path.join(fold_dir, f"predictions_eval_fold_{fold}.csv")

    model = load_model(model_path)

    df_val = pd.read_csv(pred_csv_path)
    df_meta = df_val.merge(
        df_all_meta,
        left_on="image_name",
        right_on="img_id"
    )

    flip_counts = {f: 0 for f in FEATURE_LIST}
    entropy_delta = {f: [] for f in FEATURE_LIST}
    transitions = {f: {"c0": [], "c1": []} for f in FEATURE_LIST}

    clinical_transition = {
        f: {"b2m": 0, "m2b": 0, "b2b": 0, "m2m": 0}
        for f in FEATURE_LIST
    }

    total = 0

    for _, row in df_meta.iterrows():
        img_path = os.path.join(IMAGE_DIR, row["img_id"])

        if not os.path.exists(img_path):
            continue

        img = process_image(img_path)
        meta_dict = row.to_dict()

        mutated_rows = [meta_dict]
        for f in FEATURE_LIST:
            mutated_rows.append(mutate_feature(meta_dict, f))

        meta_batch = process_metadata_batch(mutated_rows)
        preds, probs = predict_batch(model, img, meta_batch)

        pred0 = int(preds[0])
        probs0 = probs[0]
        h0 = entropy(probs0)

        for i, f in enumerate(FEATURE_LIST, start=1):
            pred1 = int(preds[i])
            probs1 = probs[i]
            h1 = entropy(probs1)

            entropy_delta[f].append(h1 - h0)

            transitions[f]["c0"].append(pred0)
            transitions[f]["c1"].append(pred1)

            if pred0 != pred1:
                flip_counts[f] += 1

            if pred0 in benign_idx and pred1 in malig_idx:
                clinical_transition[f]["b2m"] += 1
            elif pred0 in malig_idx and pred1 in benign_idx:
                clinical_transition[f]["m2b"] += 1
            elif pred0 in benign_idx and pred1 in benign_idx:
                clinical_transition[f]["b2b"] += 1
            elif pred0 in malig_idx and pred1 in malig_idx:
                clinical_transition[f]["m2m"] += 1

        total += 1

    if total == 0:
        raise RuntimeError(f"No valid images were processed in fold {fold}.")

    flip_rate = {
        f: flip_counts[f] / total
        for f in FEATURE_LIST
    }

    entropy_mean = {
        f: float(np.mean(entropy_delta[f])) if len(entropy_delta[f]) > 0 else 0.0
        for f in FEATURE_LIST
    }

    clinical_rate = {}
    for f in FEATURE_LIST:
        clinical_rate[f] = {
            "b2m": clinical_transition[f]["b2m"] / total,
            "m2b": clinical_transition[f]["m2b"] / total,
            "b2b": clinical_transition[f]["b2b"] / total,
            "m2m": clinical_transition[f]["m2m"] / total,
        }

    print(f"Fold {fold} finished | processed images: {total}")

    return flip_rate, entropy_mean, transitions, clinical_rate


# ==========================================================
# MAIN
# ==========================================================
def main():
    flip_all = []
    entropy_all = []
    clinical_all = []
    per_fold_rows = []

    global_trans = {f: {"c0": [], "c1": []} for f in FEATURE_LIST}

    for fold in range(1, 6):
        flip_rate, entropy_mean, transitions, clinical_rate = run_fold(fold)

        flip_all.append(flip_rate)
        entropy_all.append(entropy_mean)
        clinical_all.append(clinical_rate)

        for f in FEATURE_LIST:
            global_trans[f]["c0"] += transitions[f]["c0"]
            global_trans[f]["c1"] += transitions[f]["c1"]

            per_fold_rows.append({
                "fold": fold,
                "feature": f,
                "flip_rate": flip_rate[f],
                "mean_delta_entropy": entropy_mean[f],
                "benign_to_malignant_rate": clinical_rate[f]["b2m"],
                "malignant_to_benign_rate": clinical_rate[f]["m2b"],
                "benign_to_benign_rate": clinical_rate[f]["b2b"],
                "malignant_to_malignant_rate": clinical_rate[f]["m2m"],
            })

    # ------------------------------------------------------
    # Save per-fold metrics
    # ------------------------------------------------------
    df_per_fold = pd.DataFrame(per_fold_rows)
    df_per_fold.to_csv(
        os.path.join(OUT_DIR, "feature_metrics_by_fold.csv"),
        index=False
    )

    # ------------------------------------------------------
    # Flip summary
    # ------------------------------------------------------
    df_flip = pd.DataFrame(flip_all)

    flip_summary = pd.DataFrame({
        "feature": df_flip.columns,
        "mean_flip_rate": df_flip.mean().values,
        "std_flip_rate": df_flip.std().fillna(0.0).values
    }).sort_values("mean_flip_rate", ascending=False).reset_index(drop=True)

    flip_summary.to_csv(
        os.path.join(OUT_DIR, "flip_summary.csv"),
        index=False
    )

    # ------------------------------------------------------
    # Entropy summary
    # ------------------------------------------------------
    df_entropy = pd.DataFrame(entropy_all)

    entropy_summary = pd.DataFrame({
        "feature": df_entropy.columns,
        "mean_delta_entropy": df_entropy.mean().values,
        "std_delta_entropy": df_entropy.std().fillna(0.0).values
    }).sort_values("mean_delta_entropy", ascending=False).reset_index(drop=True)

    entropy_summary.to_csv(
        os.path.join(OUT_DIR, "entropy_summary.csv"),
        index=False
    )

    # ------------------------------------------------------
    # Clinical transition summary
    # ------------------------------------------------------
    clinical_rows = []
    for fold_idx, fold_data in enumerate(clinical_all, start=1):
        for f in FEATURE_LIST:
            clinical_rows.append({
                "fold": fold_idx,
                "feature": f,
                "b2m": fold_data[f]["b2m"],
                "m2b": fold_data[f]["m2b"],
                "b2b": fold_data[f]["b2b"],
                "m2m": fold_data[f]["m2m"],
            })

    df_clinical = pd.DataFrame(clinical_rows)
    df_clinical.to_csv(
        os.path.join(OUT_DIR, "clinical_transition_by_fold.csv"),
        index=False
    )

    clinical_summary = (
        df_clinical
        .groupby("feature")
        .agg(
            benign_to_malignant_mean=("b2m", "mean"),
            benign_to_malignant_std=("b2m", "std"),
            malignant_to_benign_mean=("m2b", "mean"),
            malignant_to_benign_std=("m2b", "std"),
            benign_to_benign_mean=("b2b", "mean"),
            benign_to_benign_std=("b2b", "std"),
            malignant_to_malignant_mean=("m2m", "mean"),
            malignant_to_malignant_std=("m2m", "std"),
        )
        .reset_index()
        .fillna(0.0)
        .sort_values("malignant_to_benign_mean", ascending=False)
        .reset_index(drop=True)
    )

    clinical_summary.to_csv(
        os.path.join(OUT_DIR, "clinical_transition_summary.csv"),
        index=False
    )

    # ------------------------------------------------------
    # Global plots
    # ------------------------------------------------------
    plot_feature_sensitivity_map(
        flip_summary,
        entropy_summary,
        os.path.join(OUT_DIR, "feature_sensitivity_map.png")
    )

    plot_entropy_variation(
        entropy_summary,
        os.path.join(OUT_DIR, "entropy_variation.png")
    )

    plot_flip_rate_variation(
        flip_summary,
        os.path.join(OUT_DIR, "flip_rate_variation.png")
    )

    plot_clinical_risk(
        clinical_summary,
        os.path.join(OUT_DIR, "clinical_transition_risk.png")
    )

    # ------------------------------------------------------
    # Transition matrices
    # ------------------------------------------------------
    matrix_dir = os.path.join(OUT_DIR, "transition_matrices")
    os.makedirs(matrix_dir, exist_ok=True)

    for f in FEATURE_LIST:
        c0 = global_trans[f]["c0"]
        c1 = global_trans[f]["c1"]

        cm = confusion_matrix(c0, c1, labels=range(K))

        pd.DataFrame(cm, index=CLASS_LIST, columns=CLASS_LIST).to_csv(
            os.path.join(matrix_dir, f"{f}_matrix.csv")
        )

        plot_transition_matrix(
            cm,
            f,
            os.path.join(matrix_dir, f"{f}_matrix.png")
        )

    print("\nFINAL SUMMARY - FLIP RATE")
    print(flip_summary)

    print("\nFINAL SUMMARY - ENTROPY")
    print(entropy_summary)

    print("\nFINAL SUMMARY - CLINICAL TRANSITIONS")
    print(clinical_summary)

    print(f"\nAnalysis finished. Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()