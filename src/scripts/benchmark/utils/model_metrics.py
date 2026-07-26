from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import numpy as np
import torch
import os


def evaluate_model(
    model,
    dataloader,
    targets,
    device,
    fold_num: int,
    base_dir: str,
    model_name: str = "None"
):
    """
    Universal evaluation for binary and multiclass classification.

    Returns:
        metrics : dict
        all_labels : np.ndarray [N]
        all_preds  : np.ndarray [N]
        all_probs  : np.ndarray [N, C]
    """

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------
    folder_name = f"{model_name}_fold_{fold_num}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    model.eval()

    all_labels = []
    all_preds  = []
    all_probs  = []

    # ------------------------------------------------------------
    # SINGLE FORWARD PASS
    # ------------------------------------------------------------
    with torch.no_grad():
        for (_, images, metadata, labels) in dataloader:
            images   = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            logits = model(images, metadata)           # [B, C]
            probs  = torch.softmax(logits, dim=1)      # [B, C]
            preds  = torch.argmax(probs, dim=1)         # [B]

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    # ------------------------------------------------------------
    # STACK
    # ------------------------------------------------------------
    all_labels = torch.cat(all_labels).numpy()
    all_preds  = torch.cat(all_preds).numpy()
    all_probs  = torch.cat(all_probs).numpy()

    # >>> SINGLE SOURCE OF TRUTH <<<
    num_classes = all_probs.shape[1]

    # Align targets with model output
    if targets is None:
        targets = np.arange(num_classes)
    else:
        targets = np.array(targets)[:num_classes]

    # ------------------------------------------------------------
    # SAVE RAW ARRAYS (AUDIT TRAIL)
    # ------------------------------------------------------------
    np.save(os.path.join(folder_path, "labels.npy"), all_labels)
    np.save(os.path.join(folder_path, "predictions.npy"), all_preds)
    np.save(os.path.join(folder_path, "probabilities.npy"), all_probs)
    np.save(os.path.join(folder_path, "targets.npy"), targets)

    # ------------------------------------------------------------
    # METRICS (DISCRETE)
    # ------------------------------------------------------------
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

    if num_classes == 2:
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall    = recall_score(all_labels, all_preds, zero_division=0)
        f1score   = f1_score(all_labels, all_preds, zero_division=0)
    else:
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1score   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # ------------------------------------------------------------
    # AUC (SAFE)
    # ------------------------------------------------------------
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))
            auc = roc_auc_score(
                y_true_bin,
                all_probs,
                average="weighted",
                multi_class="ovr"
            )
    except Exception as e:
        print(f"[WARN] AUC computation failed: {e}")
        auc = None

    # ------------------------------------------------------------
    # METRICS DICT
    # ------------------------------------------------------------
    metrics = {
        "fold": fold_num,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1score),
        "auc": None if auc is None else float(auc)
    }

    return metrics, all_labels, all_preds, all_probs
