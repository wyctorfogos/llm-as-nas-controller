import os
import sys
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

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
    "./src/results/artigo_1_GFCAM/12022026/"
    "PAD-UFES-20/unfrozen_weights/8/gfcam/"
    "model_densenet169_with_one-hot-encoder_512_with_best_architecture"
)

OUT_DIR = "./results/diagnostic_change_analysis_v3"
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving results to:", OUT_DIR)

CLASS_LIST = ["NEV", "BCC", "ACK", "SEK", "SCC", "MEL"]
K = len(CLASS_LIST)

FEATURE_LIST = [
    "age", "region", "gender", "itch", "grew", "bleed", "changed", "elevation",
    "hurt", "smoke", "drink", "pesticide", "skin_cancer_history",
    "cancer_history", "has_piped_water", "has_sewage_system"
]

NUMERICAL_COLS = ["age", "diameter_1", "diameter_2"]

BENIGN = ["NEV", "ACK", "SEK"]
MALIGNANT = ["BCC", "SCC", "MEL"]

benign_idx = [CLASS_LIST.index(c) for c in BENIGN]
malig_idx = [CLASS_LIST.index(c) for c in MALIGNANT]

BOOL_FEATURES = [
    "itch", "grew", "bleed", "changed", "elevation", "hurt",
    "smoke", "drink", "pesticide",
    "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system"
]

TRUTHY = {"true", "1", "1.0", "yes", "y", "t"}
FALSY = {"false", "0", "0.0", "no", "n", "f"}

# ==========================================================
# LOAD DATA & ENCODERS
# ==========================================================
df_all_meta = pd.read_csv(METADATA_PATH)

with open(os.path.join(ENCODER_DIR, "ohe_pad_20.pickle"), "rb") as f:
    OHE = pickle.load(f)

with open(os.path.join(ENCODER_DIR, "scaler_pad_20.pickle"), "rb") as f:
    SCALER = pickle.load(f)

OHE_FEATURE_NAMES = list(OHE.feature_names_in_)
SCALER_FEATURE_NAMES = list(SCALER.feature_names_in_)

# ==========================================================
# IMAGE PROCESSING
# ==========================================================
def load_image_transform(size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


IMG_TRANSFORM = load_image_transform()


def process_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)


# ==========================================================
# METADATA PROCESSING
# ==========================================================
def normalize_bool_string(value: object) -> str:
    s = str(value).strip().lower()
    if s in TRUTHY:
        return "True"
    if s in FALSY:
        return "False"
    return "False"


def process_metadata_row(row_dict: dict) -> torch.Tensor:
    """
    Usa exatamente as colunas vistas no fit do OHE e do SCALER.
    Ignora colunas extras vindas do merge com predictions CSV.
    """
    df = pd.DataFrame([row_dict])

    for col in OHE_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = "unknown"

    cat_df = df[OHE_FEATURE_NAMES].astype(str)

    for col in SCALER_FEATURE_NAMES:
        if col not in df.columns:
            df[col] = -1

    num_df = (
        df[SCALER_FEATURE_NAMES]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-1)
    )

    cat = OHE.transform(cat_df)
    num = SCALER.transform(num_df)
    processed = np.hstack([cat, num])

    return torch.tensor(processed, dtype=torch.float32).to(DEVICE)


# ==========================================================
# MODEL
# ==========================================================
def load_model(model_path: str):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=K,
        device=DEVICE,
        cnn_model_name="densenet169",
        text_model_name="one-hot-encoder",
        vocab_size=91,
        num_heads=8,
        attention_mecanism="gfcam",
        n=2,
        unfreeze_weights="unfrozen_weights",
    )

    ckpt = torch.load(model_path, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()

    return model


# ==========================================================
# MUTATE FEATURE
# ==========================================================
def mutate_metadata(row: dict, feature: str) -> dict:
    r = copy.deepcopy(row)

    if feature in BOOL_FEATURES:
        current = normalize_bool_string(r.get(feature, "False"))
        r[feature] = "False" if current == "True" else "True"
        return r

    if feature == "age":
        val = pd.to_numeric(r.get("age", np.nan), errors="coerce")
        if pd.isna(val):
            r["age"] = 80
        else:
            r["age"] = int(np.clip(val + 20, 0, 100))
        return r

    if feature == "gender":
        g = str(r.get("gender", "")).strip().upper()
        if g == "FEMALE":
            r["gender"] = "MALE"
        elif g == "MALE":
            r["gender"] = "FEMALE"
        else:
            r["gender"] = "MALE"
        return r

    if feature == "region":
        reg = str(r.get("region", "")).strip().upper()
        if reg == "FACE":
            r["region"] = "FOREARM"
        elif reg == "FOREARM":
            r["region"] = "FACE"
        else:
            r["region"] = "FACE"
        return r

    return r


# ==========================================================
# PREDICTION
# ==========================================================
def predict_class(model, img_t: torch.Tensor, meta_t: torch.Tensor) -> int:
    with torch.no_grad():
        logits = model(img_t, meta_t)
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
    return pred


# ==========================================================
# PLOT STYLE
# ==========================================================
def setup_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ==========================================================
# BAR PLOT (MEAN + STD)
# ==========================================================
def dual_bar_plot(df: pd.DataFrame, mean_col: str, std_col: str,
                  title: str, out_file: str):
    setup_style()

    df_plot = df.sort_values(mean_col, ascending=True).reset_index(drop=True)

    y = np.arange(len(df_plot))
    h = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        y - h / 2,
        df_plot[mean_col],
        height=h,
        color="#1f77b4",
        label="Mean",
    )

    ax.barh(
        y + h / 2,
        df_plot[std_col],
        height=h,
        color="#9ecae1",
        label="STD",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["feature"])
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=False)

    xmax = ax.get_xlim()[1]
    offset = max(0.005, xmax * 0.01)

    for v, yy in zip(df_plot[mean_col], y - h / 2):
        ax.text(v + offset, yy, f"{v:.3f}", va="center", ha="left", fontsize=8)

    for v, yy in zip(df_plot[std_col], y + h / 2):
        ax.text(v + offset, yy, f"{v:.3f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_file, dpi=400, bbox_inches="tight")
    plt.close()


# ==========================================================
# CONFUSION / TRANSITION HEATMAPS
# ==========================================================
def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return cm / row_sums


def plot_matrix(cm: np.ndarray, title: str, out_file: str,
                x_label: str, y_label: str, normalize: bool = False):
    setup_style()

    cm_plot = normalize_confusion_matrix(cm) if normalize else cm.copy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(X=cm_plot, cmap=plt.cm.Blues)
    plt.colorbar(im)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(CLASS_LIST, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_LIST)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for i in range(K):
        for j in range(K):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_file, dpi=400, bbox_inches="tight")
    plt.close()


# ==========================================================
# RUN FOLD
# ==========================================================
def run_fold(fold: int):
    print(f"\nRunning fold {fold}")

    fold_dir = os.path.join(BASE_RESULTS_DIR, f"densenet169_fold_{fold}")
    model_path = os.path.join(fold_dir, "model.pth")
    pred_csv = os.path.join(fold_dir, f"predictions_eval_fold_{fold}.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")

    model = load_model(model_path)
    df_val = pd.read_csv(pred_csv)

    if "image_name" not in df_val.columns:
        raise ValueError(f"'image_name' column not found in {pred_csv}")

    df_meta = df_val.merge(
        df_all_meta,
        left_on="image_name",
        right_on="img_id",
        how="inner",
    ).drop_duplicates(subset=["img_id"])

    flip_counts = {f: 0 for f in FEATURE_LIST}
    fold_transitions = {f: {"c0": [], "c1": []} for f in FEATURE_LIST}

    y_true_fold = []
    y_pred_fold = []

    total_samples = 0

    for _, row in df_meta.iterrows():
        img_path = os.path.join(IMAGE_DIR, str(row["img_id"]))
        if not os.path.exists(img_path):
            continue

        img_t = process_image(img_path)
        meta_dict = row.to_dict()

        meta_t = process_metadata_row(meta_dict)
        c0 = predict_class(model, img_t, meta_t)

        diag = str(row["diagnostic"]).strip()
        if diag in CLASS_LIST:
            y_true_fold.append(CLASS_LIST.index(diag))
            y_pred_fold.append(c0)

        for f in FEATURE_LIST:
            mutated = mutate_metadata(meta_dict, f)
            meta_mut = process_metadata_row(mutated)
            c1 = predict_class(model, img_t, meta_mut)

            fold_transitions[f]["c0"].append(c0)
            fold_transitions[f]["c1"].append(c1)

            if c1 != c0:
                flip_counts[f] += 1

        total_samples += 1

    if total_samples == 0:
        raise RuntimeError(f"Fold {fold}: no valid samples found.")

    flip_rates_fold = {
        f: flip_counts[f] / float(total_samples)
        for f in FEATURE_LIST
    }

    b2m_fold = {}
    m2b_fold = {}

    for f in FEATURE_LIST:
        c0_list = fold_transitions[f]["c0"]
        c1_list = fold_transitions[f]["c1"]

        tm = confusion_matrix(c0_list, c1_list, labels=range(K))

        benign_total = tm[benign_idx, :].sum()
        malig_total = tm[malig_idx, :].sum()

        b2m_num = tm[np.ix_(benign_idx, malig_idx)].sum()
        m2b_num = tm[np.ix_(malig_idx, benign_idx)].sum()

        b2m_fold[f] = float(b2m_num / benign_total) if benign_total > 0 else 0.0
        m2b_fold[f] = float(m2b_num / malig_total) if malig_total > 0 else 0.0

    return flip_rates_fold, b2m_fold, m2b_fold, fold_transitions, y_true_fold, y_pred_fold


# ==========================================================
# MAIN
# ==========================================================
def main():
    warnings.filterwarnings("ignore")

    all_flip_rates = []
    all_b2m_rates = []
    all_m2b_rates = []

    global_y_true = []
    global_y_pred = []

    global_feature_transitions = {
        f: {"c0": [], "c1": []}
        for f in FEATURE_LIST
    }

    for fold in range(1, 6):
        flip_fold, b2m_fold, m2b_fold, fold_transitions, y_t, y_p = run_fold(fold)

        all_flip_rates.append(flip_fold)
        all_b2m_rates.append(b2m_fold)
        all_m2b_rates.append(m2b_fold)

        global_y_true.extend(y_t)
        global_y_pred.extend(y_p)

        for f in FEATURE_LIST:
            global_feature_transitions[f]["c0"].extend(fold_transitions[f]["c0"])
            global_feature_transitions[f]["c1"].extend(fold_transitions[f]["c1"])

    # ======================================================
    # FLIP RATE SUMMARY
    # ======================================================
    df_flip = pd.DataFrame(all_flip_rates)
    flip_summary = pd.DataFrame({
        "feature": df_flip.columns,
        "mean_flip_rate": df_flip.mean(axis=0).values,
        "std_flip_rate": df_flip.std(axis=0).values,
    })
    flip_summary.to_csv(os.path.join(OUT_DIR, "flip_summary.csv"), index=False)

    dual_bar_plot(
        df=flip_summary,
        mean_col="mean_flip_rate",
        std_col="std_flip_rate",
        title="Diagnostic Change Rate (Mean vs STD across folds)",
        out_file=os.path.join(OUT_DIR, "flip_rate_mean_std.png"),
    )

    # ======================================================
    # B2M SUMMARY
    # ======================================================
    df_b2m = pd.DataFrame(all_b2m_rates)
    b2m_summary = pd.DataFrame({
        "feature": df_b2m.columns,
        "mean_b2m": df_b2m.mean(axis=0).values,
        "std_b2m": df_b2m.std(axis=0).values,
    })
    b2m_summary.to_csv(
        os.path.join(OUT_DIR, "benign_to_malignant_summary.csv"),
        index=False
    )

    dual_bar_plot(
        df=b2m_summary,
        mean_col="mean_b2m",
        std_col="std_b2m",
        title="Benign → Malignant Diagnostic Change (Mean vs STD)",
        out_file=os.path.join(OUT_DIR, "benign_to_malignant_mean_std.png"),
    )

    # ======================================================
    # M2B SUMMARY
    # ======================================================
    df_m2b = pd.DataFrame(all_m2b_rates)
    m2b_summary = pd.DataFrame({
        "feature": df_m2b.columns,
        "mean_m2b": df_m2b.mean(axis=0).values,
        "std_m2b": df_m2b.std(axis=0).values,
    })
    m2b_summary.to_csv(
        os.path.join(OUT_DIR, "malignant_to_benign_summary.csv"),
        index=False
    )

    dual_bar_plot(
        df=m2b_summary,
        mean_col="mean_m2b",
        std_col="std_m2b",
        title="Malignant → Benign Diagnostic Change (Mean vs STD)",
        out_file=os.path.join(OUT_DIR, "malignant_to_benign_mean_std.png"),
    )

    # ======================================================
    # BASELINE CONFUSION MATRIX
    # ======================================================
    if len(global_y_true) > 0 and len(global_y_pred) > 0:
        cm_base = confusion_matrix(global_y_true, global_y_pred, labels=range(K))

        pd.DataFrame(cm_base, index=CLASS_LIST, columns=CLASS_LIST).to_csv(
            os.path.join(OUT_DIR, "baseline_confusion_matrix.csv")
        )

        plot_matrix(
            cm=cm_base,
            title="Baseline Confusion Matrix",
            out_file=os.path.join(OUT_DIR, "baseline_confusion_matrix.png"),
            x_label="Predicted",
            y_label="True",
            normalize=False
        )

        plot_matrix(
            cm=cm_base,
            title="Baseline Confusion Matrix (Normalized)",
            out_file=os.path.join(OUT_DIR, "baseline_confusion_matrix_normalized.png"),
            x_label="Predicted",
            y_label="True",
            normalize=True
        )

    # ======================================================
    # FEATURE TRANSITION MATRICES
    # ======================================================
    feature_matrix_dir = os.path.join(OUT_DIR, "feature_transition_matrices")
    os.makedirs(feature_matrix_dir, exist_ok=True)

    for f in FEATURE_LIST:
        c0 = global_feature_transitions[f]["c0"]
        c1 = global_feature_transitions[f]["c1"]

        if len(c0) == 0:
            continue

        cm_feature = confusion_matrix(c0, c1, labels=range(K))
        cm_feature_norm = normalize_confusion_matrix(cm_feature)

        pd.DataFrame(cm_feature, index=CLASS_LIST, columns=CLASS_LIST).to_csv(
            os.path.join(feature_matrix_dir, f"{f}_transition_matrix.csv")
        )

        pd.DataFrame(cm_feature_norm, index=CLASS_LIST, columns=CLASS_LIST).to_csv(
            os.path.join(feature_matrix_dir, f"{f}_transition_matrix_normalized.csv")
        )

        plot_matrix(
            cm=cm_feature,
            title=f"Prediction Transition Matrix - {f}",
            out_file=os.path.join(feature_matrix_dir, f"{f}_transition_matrix.png"),
            x_label="Mutated prediction",
            y_label="Original prediction",
            normalize=False
        )

        plot_matrix(
            cm=cm_feature,
            title=f"Prediction Transition Matrix - {f} (Normalized)",
            out_file=os.path.join(feature_matrix_dir, f"{f}_transition_matrix_normalized.png"),
            x_label="Mutated prediction",
            y_label="Original prediction",
            normalize=True
        )

    print("\n✔ Analysis finished.")
    print("Results saved in:", OUT_DIR)


if __name__ == "__main__":
    main()