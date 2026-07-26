import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


def save_model_and_metrics(
    model,
    metrics: dict,
    model_name: str,
    save_to_disk: bool,
    base_dir: str,
    fold_num: int,
    all_labels: np.ndarray,
    all_predictions: np.ndarray,
    all_probabilities: np.ndarray,
    targets: list,
    data_val: str = "val",
    train_losses: np.ndarray = None,
    val_losses: np.ndarray = None
):
    """
    FINAL, ROBUST VERSION (Binary + Multiclass Safe)

    Assumptions:
    - all_labels        : shape [N]
    - all_predictions  : shape [N]
    - all_probabilities: shape [N, C]
    """

    # ------------------------------------------------------------
    # PREP
    # ------------------------------------------------------------
    all_labels = np.asarray(all_labels)
    all_predictions = np.asarray(all_predictions)
    all_probabilities = np.asarray(all_probabilities)

    # 🔑 Real number of classes comes from model output
    num_classes = all_probabilities.shape[1]

    # 🔒 Make targets compatible with model
    if targets is None:
        targets = [str(i) for i in range(num_classes)]
    else:
        targets = list(targets)[:num_classes]

    folder_name = f"{model_name}_fold_{fold_num}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # ------------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------------
    if save_to_disk:
        model_path = os.path.join(folder_path, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✔ Model saved at: {model_path}")

    # ------------------------------------------------------------
    # SAVE METRICS CSV (APPEND)
    # ------------------------------------------------------------
    metrics_file = os.path.join(base_dir, "model_metrics.csv")
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"✔ Metrics saved at: {metrics_file}")

    # ------------------------------------------------------------
    # CONFUSION MATRIX (NORMALIZED)
    # ------------------------------------------------------------
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        normalize="true"
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    disp = ConfusionMatrixDisplay(cm, display_labels=targets)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format=".3f")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder_path, "confusion_matrix.png"),
        dpi=400,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # ROC CURVE
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 9))

    if num_classes == 2:
        # Binary ROC
        y_true = all_labels
        y_scores = all_probabilities[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")

    else:
        # Multiclass One-vs-Rest ROC
        y_true_bin = label_binarize(
            all_labels,
            classes=np.arange(num_classes)
        )

        for i in range(num_classes):
            label = targets[i]
            fpr, tpr, _ = roc_curve(
                y_true_bin[:, i],
                all_probabilities[:, i]
            )
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")

    # Chance line
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(
        os.path.join(folder_path, "roc_curve.png"),
        dpi=400,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # LOSS CURVES
    # ------------------------------------------------------------
    if train_losses is not None and val_losses is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label="Train Loss", marker="o")
        plt.plot(val_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)

        loss_plot_path = os.path.join(
            base_dir, f"loss_curve_fold_{fold_num}.png"
        )
        plt.savefig(loss_plot_path, dpi=400, bbox_inches="tight")
        plt.close()

        print(f"✔ Loss curve saved at: {loss_plot_path}")

    return folder_path
