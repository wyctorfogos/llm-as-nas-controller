import os
import sys
import pickle
import traceback
import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Ajuste conforme seu ambiente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from benchmark.models import multimodalIntraInterModal


# ==========================================================
# 1. CONSTANTES E CONFIGURAÇÕES
# ==========================================================

MAPPING = {0: "ACK", 1: "BCC", 2: "MEL", 3: "NEV", 4: "SCC", 5: "SEK"}
LABELS_LIST = [MAPPING[i] for i in range(6)]

MISSING_RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

METADATA_FILE = "./data/PAD-UFES-20/metadata.csv"
IMAGE_ROOT = "./data/PAD-UFES-20/images"
OHE_PATH = "./data/preprocess_data/ohe_pad_20.pickle"
SCALER_PATH = "./data/preprocess_data/scaler_pad_20.pickle"

PAD_COLUMNS = [
    "patient_id", "lesion_id", "smoke", "drink", "background_father",
    "background_mother", "age", "pesticide", "gender", "skin_cancer_history",
    "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick",
    "region", "diameter_1", "diameter_2", "diagnostic", "itch", "grew",
    "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"
]

NUMERICAL_COLS = ["age", "diameter_1", "diameter_2"]
CATEGORICAL_COLS = [
    c for c in PAD_COLUMNS
    if c not in NUMERICAL_COLS + ["patient_id", "lesion_id", "img_id", "biopsed", "diagnostic"]
]
DROP_COLS = ["patient_id", "lesion_id", "img_id", "biopsed", "diagnostic"]


# ==========================================================
# 2. FUNÇÕES AUXILIARES
# ==========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def log_message(log_file: str, message: str, print_also: bool = True):
    ensure_dir(os.path.dirname(log_file))
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    if print_also:
        print(message)


def log_exception(log_file: str, prefix: str, exc: Exception, print_also: bool = True):
    msg = f"{prefix}: {repr(exc)}"
    trace = traceback.format_exc()
    ensure_dir(os.path.dirname(log_file))
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.write(trace + "\n")
        f.write("=" * 100 + "\n")
    if print_also:
        print(msg)
        print(trace)


def _strip_module_prefix(state_dict):
    """Remove o prefixo 'module.' de modelos salvos com DataParallel."""
    if not isinstance(state_dict, dict):
        return state_dict

    keys = list(state_dict.keys())
    if len(keys) > 0 and keys[0].startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("EMPTY")
    df = df.replace(r"^\s*$", "EMPTY", regex=True)
    df = df.replace([" ", "  ", "NÃO  ENCONTRADO"], "EMPTY")
    df = df.replace("BRASIL", "BRAZIL")
    return df


def parse_csv_line_to_cols(sample, columns: list) -> pd.DataFrame:
    """Garante que a entrada tenha todas as colunas na ordem certa."""
    if isinstance(sample, pd.DataFrame):
        sample = sample.copy()
        for col in columns:
            if col not in sample.columns:
                sample[col] = "EMPTY"
        return sample[columns].copy()

    if isinstance(sample, str):
        parts = sample.split(",")
    else:
        parts = list(sample)

    if len(parts) < len(columns):
        parts = parts + [""] * (len(columns) - len(parts))
    else:
        parts = parts[:len(columns)]

    return pd.DataFrame([parts], columns=columns)


def process_metadata_pad20(df_raw, ohe, scaler, device):
    df = parse_csv_line_to_cols(df_raw, PAD_COLUMNS)
    df = clean_metadata(df)

    features = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

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

    # Mantém o mesmo tamanho esperado pelo checkpoint
    target_size = 91
    if processed.shape[1] < target_size:
        diff = target_size - processed.shape[1]
        padding = np.zeros((processed.shape[0], diff))
        processed = np.hstack([processed, padding])
    elif processed.shape[1] > target_size:
        processed = processed[:, :target_size]

    return torch.tensor(processed, dtype=torch.float32).to(device)


def simulate_missing_metadata(df, missing_rate, numerical_cols, categorical_cols, seed=42):
    df = df.copy()
    rng = np.random.default_rng(seed)

    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(object)

    all_features = numerical_cols + categorical_cols
    mask = rng.random((len(df), len(all_features))) < (1 - missing_rate)

    for j, col in enumerate(all_features):
        if col not in df.columns:
            continue

        if col in numerical_cols:
            df.loc[~mask[:, j], col] = np.nan
        else:
            df.loc[~mask[:, j], col] = "EMPTY"

    return df


def build_model(cnn_name, attention, device):
    model = multimodalIntraInterModal.MultimodalModel(
        num_classes=6,
        device=device,
        cnn_model_name=cnn_name,
        text_model_name="one-hot-encoder",
        vocab_size=91,
        num_heads=8,
        attention_mecanism=attention,
        n=2,
        unfreeze_weights="unfrozen_weights"
    )
    return model


def safe_load_checkpoint(model, model_path, device, log_file):
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = _strip_module_prefix(state_dict)

    incompatible = model.load_state_dict(state_dict, strict=False)
    log_message(log_file, f"✅ Checkpoint carregado com strict=False: {model_path}")

    if hasattr(incompatible, "missing_keys") and incompatible.missing_keys:
        log_message(log_file, f"⚠️ Missing keys: {incompatible.missing_keys}")

    if hasattr(incompatible, "unexpected_keys") and incompatible.unexpected_keys:
        log_message(log_file, f"⚠️ Unexpected keys: {incompatible.unexpected_keys}")

    return model


def safe_open_image(image_path):
    with Image.open(image_path) as img:
        return img.convert("RGB")


def get_true_label_from_row(row):
    """
    Prioriza um rótulo numérico, se existir no CSV.
    Caso não exista, cai para o diagnostic textual.
    """
    numeric_candidates = ["label", "true_label", "target", "y_true", "labels"]

    for col in numeric_candidates:
        if col in row.index and pd.notna(row[col]):
            try:
                return int(row[col])
            except Exception:
                pass

    diagnostic = row["diagnostic"]
    if diagnostic not in LABELS_LIST:
        raise ValueError(f"Rótulo desconhecido: {diagnostic}")

    return LABELS_LIST.index(diagnostic)


def compute_metrics_like_training(y_true, y_pred, y_prob, targets=None):
    """
    Replica a mesma forma de validação do treino.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    num_classes = y_prob.shape[1]

    if targets is None:
        targets = np.arange(num_classes)
    else:
        targets = np.array(targets)[:num_classes]

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    if num_classes == 2:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1score = f1_score(y_true, y_pred, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1score = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
            auc = roc_auc_score(
                y_true_bin,
                y_prob,
                average="weighted",
                multi_class="ovr"
            )
    except Exception as e:
        print(f"[WARN] AUC computation failed: {e}")
        auc = None

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1score),
        "auc": None if auc is None else float(auc)
    }

    return metrics


# ==========================================================
# 3. EXECUÇÃO DO BENCHMARK
# ==========================================================

if __name__ == "__main__":
    print(f"🖥️ Usando dispositivo: {DEVICE}")

    with open(OHE_PATH, "rb") as f:
        shared_ohe = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        shared_scaler = pickle.load(f)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ])

    meta_all = pd.read_csv(METADATA_FILE)

    BACKBONES = [
        "davit_tiny.msft_in1k",
        "mvitv2_small.fb_in1k",
        "caformer_b36.sail_in22k_ft_in1k",
        "mobilenet-v2",
        "densenet169",
        "resnet-50",
        "efficientnet-b0"
    ]

    MECHANISMS =  ["no-metadata", "concatenation", "metablock", "crossattention", "att-intramodal+residual+cross-attention-metadados"] 

    summary_output_dir = "./src/results/testes-da-implementacao-final_2/02042026-WITH-LN--METHOD-CONFIG-COMPARISON/unfrozen_weights/8/summary"
    ensure_dir(summary_output_dir)

    for cnn_name in BACKBONES:
        for attention in MECHANISMS:
            print(f"\n⚙️ EXPERIMENTO: {cnn_name} | {attention}")

            base_results_dir = (
                f"./src/results/testes-da-implementacao-final_2/02042026-WITH-LN--METHOD-CONFIG-COMPARISON/"
                f"PAD-UFES-20/unfrozen_weights/8/{attention}/"
                f"model_{cnn_name}_with_one-hot-encoder_512_with_best_architecture"
            )

            experiment_log_file = os.path.join(
                summary_output_dir,
                f"errors_{attention}_{cnn_name}.log"
            )

            all_folds_data = []

            for fold_idx in range(1, 6):
                print(f"\n🔁 Processando Fold {fold_idx}")

                fold_folder = f"{cnn_name}_fold_{fold_idx}"
                fold_path = os.path.join(base_results_dir, fold_folder)

                model_path = os.path.join(fold_path, "model.pth")
                if not os.path.exists(model_path):
                    model_path = os.path.join(fold_path, "best-model", "best_model.pt")

                if not os.path.exists(model_path):
                    log_message(
                        experiment_log_file,
                        f"⚠️ Fold {fold_idx}: modelo não encontrado em {fold_path}"
                    )
                    continue

                # ------------------------------------------------------
                # 3.1 Carregamento do modelo
                # ------------------------------------------------------
                try:
                    model = build_model(cnn_name, attention, DEVICE)
                    model = safe_load_checkpoint(model, model_path, DEVICE, experiment_log_file)
                    model.to(DEVICE).eval()

                except Exception as e:
                    log_exception(
                        experiment_log_file,
                        f"❌ Fold {fold_idx}: erro ao carregar o modelo",
                        e
                    )
                    continue

                # ------------------------------------------------------
                # 3.2 Leitura do CSV de predições e merge
                # ------------------------------------------------------
                preds_csv = os.path.join(fold_path, f"predictions_eval_fold_{fold_idx}.csv")

                try:
                    if not os.path.exists(preds_csv):
                        raise FileNotFoundError(f"CSV não encontrado: {preds_csv}")

                    preds_fold = pd.read_csv(preds_csv)

                    if preds_fold.empty:
                        raise ValueError(f"CSV vazio: {preds_csv}")

                    right_key = "image_name" if "image_name" in preds_fold.columns else "img_id"

                    if right_key not in preds_fold.columns:
                        raise KeyError(
                            f"Nem 'image_name' nem 'img_id' encontrados no CSV {preds_csv}"
                        )

                    df_test = pd.merge(meta_all, preds_fold, left_on="img_id", right_on=right_key)

                    if df_test.empty:
                        raise ValueError(
                            f"Merge vazio entre metadata.csv e predictions_eval_fold_{fold_idx}.csv"
                        )

                except Exception as e:
                    log_exception(
                        experiment_log_file,
                        f"❌ Fold {fold_idx}: erro ao preparar df_test",
                        e
                    )
                    continue

                # ------------------------------------------------------
                # 3.3 Inferência por missing rate
                # ------------------------------------------------------
                for rate in MISSING_RATES:
                    y_true, y_pred, y_prob = [], [], []
                    skipped_samples = 0

                    print(f"   🧪 Missing Rate: {rate}")

                    try:
                        df_test_missing = simulate_missing_metadata(
                            df=df_test,
                            missing_rate=rate,
                            numerical_cols=NUMERICAL_COLS,
                            categorical_cols=CATEGORICAL_COLS,
                            seed=fold_idx + int(rate * 1000)
                        )
                    except Exception as e:
                        log_exception(
                            experiment_log_file,
                            f"❌ Fold {fold_idx} | Rate {rate}: erro ao simular missing metadata",
                            e
                        )
                        continue

                    for idx, row in df_test_missing.iterrows():
                        try:
                            sample_data = {}
                            for col in PAD_COLUMNS:
                                if col in row.index:
                                    sample_data[col] = row[col]
                                else:
                                    sample_data[col] = "EMPTY"

                            sample_df = pd.DataFrame([sample_data])

                            if "img_id" not in row or pd.isna(row["img_id"]):
                                raise ValueError("img_id ausente ou nulo")

                            img_path = os.path.join(IMAGE_ROOT, str(row["img_id"]))

                            if not os.path.exists(img_path):
                                raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

                            img = safe_open_image(img_path)
                            img_np = np.array(img)
                            img_t = transform(image=img_np)["image"].unsqueeze(0).to(DEVICE)

                            meta_t = process_metadata_pad20(
                                sample_df,
                                shared_ohe,
                                shared_scaler,
                                DEVICE
                            )

                            true_label = get_true_label_from_row(row)

                            with torch.no_grad():
                                output = model(img_t, meta_t)
                                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                                pred_idx = int(np.argmax(probs))

                            y_true.append(true_label)
                            y_pred.append(pred_idx)
                            y_prob.append(probs)

                        except Exception as e:
                            skipped_samples += 1
                            log_message(
                                experiment_log_file,
                                (
                                    f"⚠️ Fold {fold_idx} | Rate {rate} | Sample idx {idx} "
                                    f"| img_id={row.get('img_id', 'N/A')} | Erro: {repr(e)}"
                                )
                            )
                            continue

                    # --------------------------------------------------
                    # 3.4 Métricas no mesmo padrão do treino
                    # --------------------------------------------------
                    if len(y_true) == 0:
                        log_message(
                            experiment_log_file,
                            f"⚠️ Fold {fold_idx} | Rate {rate}: nenhuma amostra válida."
                        )
                        continue

                    try:
                        y_true = np.array(y_true)
                        y_pred = np.array(y_pred)
                        y_prob = np.array(y_prob)

                        np.save(os.path.join(fold_path, f"labels_missing_{rate}.npy"), y_true)
                        np.save(os.path.join(fold_path, f"predictions_missing_{rate}.npy"), y_pred)
                        np.save(os.path.join(fold_path, f"probabilities_missing_{rate}.npy"), y_prob)
                        np.save(
                            os.path.join(fold_path, f"targets_missing_{rate}.npy"),
                            np.arange(y_prob.shape[1])
                        )

                        metrics = compute_metrics_like_training(
                            y_true=y_true,
                            y_pred=y_pred,
                            y_prob=y_prob,
                            targets=np.arange(y_prob.shape[1])
                        )

                        acc = metrics["accuracy"]
                        bacc = metrics["balanced_accuracy"]
                        precision = metrics["precision"]
                        recall = metrics["recall"]
                        f1 = metrics["f1_score"]
                        auc = np.nan if metrics["auc"] is None else metrics["auc"]

                        all_folds_data.append({
                            "fold": fold_idx,
                            "missing_rate": rate,
                            "accuracy": acc,
                            "balanced_acc": bacc,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                            "auc": auc,
                            "valid_samples": len(y_true),
                            "skipped_samples": skipped_samples
                        })

                        print(
                            f"   ➔ Rate {rate} | "
                            f"ACC: {acc:.4f} | "
                            f"BAcc: {bacc:.4f} | "
                            f"Precision: {precision:.4f} | "
                            f"Recall: {recall:.4f} | "
                            f"F1: {f1:.4f} | "
                            f"AUC: {auc:.4f} | "
                            f"Valid: {len(y_true)} | "
                            f"Skipped: {skipped_samples}"
                        )

                    except Exception as e:
                        log_exception(
                            experiment_log_file,
                            f"❌ Fold {fold_idx} | Rate {rate}: erro ao calcular métricas",
                            e
                        )
                        continue

                # limpeza opcional de memória
                try:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            # ----------------------------------------------------------
            # 3.5 Resumo final
            # ----------------------------------------------------------
            if all_folds_data:
                try:
                    res_df = pd.DataFrame(all_folds_data)

                    raw_out_path = os.path.join(
                        summary_output_dir,
                        f"raw_results_{attention}_{cnn_name}.csv"
                    )
                    res_df.to_csv(raw_out_path, index=False)

                    summary = res_df.groupby("missing_rate").agg({
                        "accuracy": ["mean", "std"],
                        "balanced_acc": ["mean", "std"],
                        "precision": ["mean", "std"],
                        "recall": ["mean", "std"],
                        "f1_score": ["mean", "std"],
                        "auc": ["mean", "std"],
                        "valid_samples": ["mean"],
                        "skipped_samples": ["mean"]
                    }).reset_index()

                    summary.columns = [
                        "missing_rate",
                        "acc_mean", "acc_std",
                        "bacc_mean", "bacc_std",
                        "precision_mean", "precision_std",
                        "recall_mean", "recall_std",
                        "f1_mean", "f1_std",
                        "auc_mean", "auc_std",
                        "valid_samples_mean",
                        "skipped_samples_mean"
                    ]

                    summary["backbone"] = cnn_name
                    summary["mechanism"] = attention

                    out_path = os.path.join(
                        summary_output_dir,
                        f"summary_{attention}_{cnn_name}.csv"
                    )
                    summary.to_csv(out_path, index=False)

                    print(f"\n📊 Resumo final gerado com sucesso!")
                    print(f"📍 Raw salvo em: {raw_out_path}")
                    print(f"📍 Summary salvo em: {out_path}")

                except Exception as e:
                    log_exception(
                        experiment_log_file,
                        f"❌ Erro ao salvar resumo final de {cnn_name} | {attention}",
                        e
                    )
            else:
                log_message(
                    experiment_log_file,
                    f"⚠️ Nenhum resultado válido para {cnn_name} | {attention}"
                )