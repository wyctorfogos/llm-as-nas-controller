import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# ============================================================
# MAIN
# ============================================================
def evaluate_from_csv_golden(CSV_PATTERN:str, OUTPUT_DIR:str, CLASSES:list, NORMALIZE_CM = True):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------
    # LOAD CSVs
    # ------------------------------------------------------------
    files = sorted(glob.glob(CSV_PATTERN))
    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found for pattern: {CSV_PATTERN}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # ------------------------------------------------------------
    # SANITY CHECKS (CRÍTICO)
    # ------------------------------------------------------------
    required_cols = {"label_idx", "prediction_idx"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError("CSV must contain label_idx and prediction_idx")

    prob_cols = [f"prob_{c}" for c in CLASSES]
    for c in prob_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing probability column: {c}")

    # ------------------------------------------------------------
    # USE ORIGINAL INDICES (EXACTLY AS TRAINING)
    # ------------------------------------------------------------
    y_true = df["label_idx"].values.astype(int)
    y_pred = df["prediction_idx"].values.astype(int)

    y_scores = df[prob_cols].values

    # Softmax defensivo (igual ao treino)
    if not np.allclose(y_scores.sum(axis=1), 1.0):
        exp_scores = np.exp(y_scores)
        y_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    n_classes = len(CLASSES)

    # ------------------------------------------------------------
    # CONFUSION MATRIX (IDÊNTICA AO TREINO)
    # ------------------------------------------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=range(n_classes),
        normalize="true" if NORMALIZE_CM else None
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".3f")

    ax.set_title("Confusion Matrix")
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        dpi=400,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # ROC CURVE (MULTICLASS – IDÊNTICA AO TREINO)
    # ------------------------------------------------------------
    y_true_bin = label_binarize(
        y_true,
        classes=range(n_classes)
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    for i, label in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "roc_curve.png"),
        dpi=400,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("✔ Evaluation completed")
    print(f"✔ Results saved to: {OUTPUT_DIR}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # CONFIG (DEVE SER IDÊNTICO AO TREINO)
    # ============================================================
    CLASSES = [
        "AKIEC", "BCC", "BEN_OTH", "BKL", "DF",
        "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
    ]

    CSV_PATTERN = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/MILK10k/dermoscopic/unfrozen_weights/8/att-intramodal+residual+cross-attention-metadados/model_caformer_b36.sail_in22k_ft_in1k_with_one-hot-encoder_512_with_best_architecture/caformer_b36.sail_in22k_ft_in1k_fold_1/predictions_eval_fold_1.csv"
    OUTPUT_DIR  = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/MILK10k/dermoscopic/unfrozen_weights/8/att-intramodal+residual+cross-attention-metadados/model_caformer_b36.sail_in22k_ft_in1k_with_one-hot-encoder_512_with_best_architecture/caformer_b36.sail_in22k_ft_in1k_fold_1"
    NORMALIZE_CM = True

    evaluate_from_csv_golden(CSV_PATTERN=CSV_PATTERN, OUTPUT_DIR=OUTPUT_DIR, CLASSES=CLASSES)
