import numpy as np
import torch
import os
import pandas as pd


def save_image_prediction_in_evaluation_by_fold(
    file_folder_path: str,
    fold_num: int,
    image_names,
    labels,
    preds,
    probs,
    targets
):
    try:
        image_names = list(image_names)
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
        preds  = preds.cpu().numpy()  if isinstance(preds, torch.Tensor)  else np.array(preds)
        probs  = probs.cpu().numpy()  if isinstance(probs, torch.Tensor)  else np.array(probs)

        n = len(image_names)
        if not all(len(x) == n for x in [labels, preds, probs]):
            print(f"Tamanhos incompatíveis:")
            print(f"image_names={len(image_names)}, labels={len(labels)}, preds={len(preds)}, probs={len(probs)}")
            return

        # -----------------------------
        # Number of classes from model
        # -----------------------------
        num_classes = probs.shape[1]

        # Use only valid targets
        safe_targets = list(targets)[:num_classes]

        # -----------------------------
        # Map labels to names safely
        # -----------------------------
        label_names = [safe_targets[i] if i < len(safe_targets) else str(i) for i in labels]
        pred_names  = [safe_targets[i] if i < len(safe_targets) else str(i) for i in preds]

        # -----------------------------
        # Build dataframe
        # -----------------------------
        df = pd.DataFrame({
            "image_name": image_names,
            "label_idx": labels,
            "label": label_names,
            "prediction_idx": preds,
            "prediction": pred_names
        })

        # -----------------------------
        # Add probabilities safely
        # -----------------------------
        for class_idx in range(num_classes):
            class_name = safe_targets[class_idx]
            df[f"prob_{class_name}"] = probs[:, class_idx]

        # -----------------------------
        # Save
        # -----------------------------
        csv_path = os.path.join(file_folder_path, f"predictions_eval_fold_{fold_num}.csv")
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=write_header, index=False)

    except Exception as e:
        print(f"Erro ao tentar salvar as predições! Erro: {e}\n")


def model_val_predictions(
    model,
    dataloader,
    targets,
    device,
    fold_num,
    base_dir,
    model_name=None
):
    folder_name = f"{model_name}_fold_{fold_num}"
    folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for (image_names, images, metadata, labels) in dataloader:
            images   = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            outputs = model(images, metadata)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            save_image_prediction_in_evaluation_by_fold(
                file_folder_path=folder_path,
                fold_num=fold_num,
                targets=targets,
                image_names=image_names,
                preds=preds,
                probs=probs,
                labels=labels
            )
