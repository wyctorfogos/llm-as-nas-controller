from xml.parsers.expat import model

import torch
import torch.nn as nn
import os
import csv
import sys
import json
import time
import numpy as np
import mlflow
import gc
from tqdm import tqdm
from pydantic import ValidationError
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models import skinLesionDatasetsPAD2020
from utils import model_metrics
from utils.early_stopping import EarlyStopping
from utils import load_local_variables
from utils.request_to_llm import request_to_ollama, filter_generated_response
from utils.save_model_and_metrics import save_model_and_metrics

from models.pydantic_llm_response_formats import NASConfig
from models import dynamicMultimodalmodel


def cleanup_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
# ============================================================
# Utils
# ============================================================
def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


def build_history_full(history):
    return "\n".join(
        f"Step {i+1}: BACC={h['reward']:.4f}, config={json.dumps(h['config'])}"
        for i, h in enumerate(history)
    )


def build_history_last_k(history, k=10):
    return "\n".join(
        f"Recent-{i+1}: BACC={h['reward']:.4f}, config={json.dumps(h['config'])}"
        for i, h in enumerate(history[-k:])
    )


def build_history_top_k(history, k=10):
    topk = sorted(history, key=lambda x: x["reward"], reverse=True)[:k]
    return "\n".join(
        f"Top-{i+1}: BACC={h['reward']:.4f}, config={json.dumps(h['config'])}"
        for i, h in enumerate(topk)
    )


def build_history(history, history_mode="full", k=10):
    if not history:
        return "No history yet."
    if history_mode == "full":
        return build_history_full(history)
    if history_mode == "last_k":
        return build_history_last_k(history, k)
    if history_mode == "top_k":
        return build_history_top_k(history, k)
    raise ValueError(f"Unknown HISTORY_MODE: {history_mode}")


# ============================================================
# Train process
# ============================================================
def train_process(
    config,
    num_epochs,
    num_heads,
    fold_num,
    train_loader,
    val_loader,
    targets,
    model,
    device,
    class_weights,
    common_dim,
    model_name,
    text_model_encoder,
    attention_mechanism,
    llm_model_name,
    results_folder_path
):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2
    )

    model.to(device)

    model_save_path = os.path.join(
        results_folder_path,
        f"model_{model_name}_{text_model_encoder}_{common_dim}"
    )
    os.makedirs(model_save_path, exist_ok=True)

    early_stopping = EarlyStopping(
        patience=10,
        delta=0.01,
        save_to_disk=False,
        early_stopping_metric_name="val_loss"
    )
    initial_time = time.time()
    train_losses, val_losses = [], []

    # usa variável global dataset_folder_name definida no main
    experiment_name = f"EXPERIMENTOS-NAS-{dataset_folder_name} -- LLM AS CONTROLLER WITH {str(HISTORY_MODE).upper()} TRAIN PROCESS HISTORY AND PYDANTIC - OPTIMIZATION TRAIN PROCESS - 20/05/2026 - COT ABLATION"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=(
            f"image_extractor_model_{model_name}_with_mechanism_"
            f"{attention_mechanism}_step_{fold_num}_num_heads_{num_heads}"
        ), nested=True
    ):
        mlflow.log_param("step", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mechanism", attention_mechanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        mlflow.log_param("criterion_type", "cross_entropy")
        mlflow.log_param("num_heads", num_heads)
        mlflow.log_param("controller_llm_model_name", llm_model_name)
        mlflow.log_param("history_mode", HISTORY_MODE)
        mlflow.log_param("search_steps", SEARCH_STEPS)
        mlflow.log_param("search_space", json.dumps(search_space))

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for _, img, meta, y in train_loader:
                img, meta, y = img.to(device), meta.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(img, meta), y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            train_losses.append(float(train_loss))
            print(f"\nTraining: Epoch {epoch}, Loss: {train_loss:.4f}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ , image, metadata, label in val_loader:
                    image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            val_losses.append(float(val_loss))
            print(f"Validation Loss: {val_loss:.4f}")

            # scheduler.step(val_loss)
            current_lr = [pg['lr'] for pg in optimizer.param_groups]
            print(f"Current Learning Rate(s): {current_lr}\n")

            metrics, _, _, _ = model_metrics.evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                fold_num=fold_num,
                targets=targets,
                base_dir=model_save_path,
                model_name=model_name
            )
            metrics["epoch"] = epoch
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)
            metrics["attention_mechanism"] = str(attention_mechanism)
            metrics["common_dim"]=int(common_dim)
            print(f"Metrics: {metrics}\n")

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=epoch + 1)
                else:
                    mlflow.log_param(metric_name, metric_value)

            val_bacc = float(metrics["balanced_accuracy"])
            scheduler.step(val_loss)
            early_stopping(val_loss=val_loss, val_bacc=val_bacc, model=model)

            mlflow.log_metric("val_bacc", val_bacc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if early_stopping.early_stop:
                break
    
    train_process_time = time.time() - initial_time
    
    # Carrega o melhor modelo encontrado
    model = early_stopping.load_best_weights(model)
    model.eval()
    # Inferência para validação com o melhor modelo
    with torch.no_grad():
        metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
            model=model, dataloader = val_loader, device=device, fold_num=fold_num, targets=targets, base_dir=model_save_path, model_name=model_name 
        )
    
    metrics["train process time"] = str(train_process_time)
    metrics["epochs"] = str(int(epoch))
    metrics["data_val"] = "val"
    metrics["epoch"] = epoch
    metrics["train_loss"] = float(train_loss)
    metrics["val_loss"] = float(val_loss)
    metrics["attention_mechanism"] = str(attention_mechanism)
    metrics["common_dim"]=int(common_dim)

    print(f"Model saved at {model_save_path}")
    
    # Salvar os dados da configuração
    folder_name = f"{model_name}_fold_{fold_num}"
    folder_path = os.path.join(model_save_path, folder_name)

    # Criar a pasta para o modelo
    os.makedirs(folder_path, exist_ok=True)

    save_model_and_metrics(
        model=model,
        metrics=metrics,
        model_name=model_name,
        base_dir=model_save_path,
        save_to_disk=True,
        fold_num=fold_num,
        all_labels=all_labels,
        all_predictions=all_predictions,
        all_probabilities=all_probs,
        targets=targets,
        data_val="val",
        train_losses=train_losses,
        val_losses=val_losses
    )
    mlflow.log_param("search_space", json.dumps(search_space))
    mlflow.log_metric("controller_reward", val_bacc, step=fold_num)
    mlflow.log_metric("dynamic_cnn_val_loss", val_loss, step=fold_num)
    mlflow.log_param(f"config_step_{fold_num}", json.dumps(config))

    # Salvar as métricas
    metrics_file = os.path.join(results_folder_path, "all_model_metrics.csv")
    file_exists = os.path.isfile(metrics_file)

    with open(metrics_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)
    
    with open(os.path.join(folder_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    del optimizer, scheduler, criterion
    del model
    cleanup_cuda()
    
    return model_save_path, metrics

# ============================================================
# NAS Pipeline
# ============================================================
def pipeline(
    dataset,
    num_metadata_features,
    num_epochs,
    batch_size,
    device,
    num_classes,
    model_name,
    num_heads,
    text_model_encoder,
    attention_mechanism,
    results_folder_path,
    SEARCH_STEPS,
    search_space,
    llm_model_name,
    history_mode="full",
    thinking=False,
    history_k=10,
    num_workers=5,
    persistent_workers=False,
    test_size=0.2,
    resume=True
):
    os.makedirs(results_folder_path, exist_ok=True)

    history_path = os.path.join(results_folder_path, "nas_history.jsonl")
    best_cfg_path = os.path.join(results_folder_path, "best_config.json")

    # ================= DATA SPLIT =================
    labels = [dataset.labels[i] for i in range(len(dataset))]

    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=labels,
        random_state=42
    )

    train_dataset = type(dataset)(
        metadata_file=dataset.metadata_file,
        img_dir=dataset.img_dir,
        size=(224, 224),
        drop_nan=dataset.is_to_drop_nan,
        bert_model_name=dataset.bert_model_name,
        image_encoder=dataset.image_encoder,
        is_train=True
    )
    train_dataset.metadata = dataset.metadata.iloc[train_idx].reset_index(drop=True)
    train_dataset.features, train_dataset.labels, train_dataset.targets = train_dataset.one_hot_encoding()

    val_dataset = type(dataset)(
        metadata_file=dataset.metadata_file,
        img_dir=dataset.img_dir,
        size=(224, 224),
        drop_nan=dataset.is_to_drop_nan,
        bert_model_name=dataset.bert_model_name,
        image_encoder=dataset.image_encoder,
        is_train=False
    )
    val_dataset.metadata = dataset.metadata.iloc[val_idx].reset_index(drop=True)
    val_dataset.features, val_dataset.labels, val_dataset.targets = val_dataset.one_hot_encoding()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=torch.cuda.is_available()
    )

    class_weights = compute_class_weights(
        [labels[i] for i in train_idx],
        num_classes
    ).to(device)

    # ================= NAS STATE =================
    history = []
    tested_configs = set()
    best_reward = -float("inf")
    best_config = None
    best_step = -1

    if resume and os.path.exists(history_path):
        print(f"Resumindo histórico de: {history_path}")

        with open(history_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)
                config = item["config"]
                reward = float(item["reward"])
                step = int(item.get("step", len(history) + 1))

                history.append({
                    "config": config,
                    "reward": reward
                })

                tested_configs.add(json.dumps(config, sort_keys=True))

                if reward > best_reward:
                    best_reward = reward
                    best_config = config
                    best_step = step

        print(f"Histórico carregado: {len(history)} arquiteturas já testadas.")
        print(f"Melhor BACC atual: {best_reward:.4f}")

    experiment_name = (
        f"NAS-{dataset_folder_name}-LLM-{llm_model_name}-"
        f"HISTORY-{history_mode.upper()}"
    )
    mlflow.set_experiment(experiment_name)

    start_step = len(history) + 1

    # ================= RUN PAI =================
    with mlflow.start_run(run_name="NAS_CONTROLLER"):

        mlflow.log_param("search_steps", SEARCH_STEPS)
        mlflow.log_param("history_mode", history_mode)
        mlflow.log_param("llm_model_name", llm_model_name)
        mlflow.log_param("search_space", json.dumps(search_space))

        for step in range(start_step, SEARCH_STEPS + 1):
            print(f"\n[STEP {step}]")

            dynamic_model = None
            metrics = None
            config_llm = None
            reward = None
            val_loss = None

            history_text = build_history(
                history,
                history_mode=history_mode,
                k=history_k
            )

            prompt = f"""
You are an AI NAS controller for skin lesion classification.

Goal: maximize Balanced Accuracy (BACC).

Search space:
{json.dumps(search_space, indent=2)}

History:
{history_text}

Based on this history, propose ONE new architecture configuration that is likely to IMPROVE BACC compared to the best so far.

You should:
- Prefer configurations similar to high-BACC ones, but with small changes.
- Avoid repeating exactly the same configuration as past attempts.
- Always keep hyperparameter values inside the given search_space.
- If the history is bad, you may explore more diverse configurations.

Return ONLY ONE JSON OBJECT, with EXACTLY these keys:
{{
  "num_blocks": <int>,
  "initial_filters": <int>,
  "kernel_size": <int>,
  "layers_per_block": <int>,
  "use_pooling": <true or false>,
  "common_dim": <int>,
  "attention_mechanism": <string>,
  "num_layers_text_fc": <int>,
  "neurons_per_layer_size_of_text_fc": <int>,
  "num_layers_fc_module": <int>,
  "neurons_per_layer_size_of_fc_module": <int>
}}
"""

            try:
                # ================= LLM CONFIG =================
                response = request_to_ollama(
                    prompt,
                    model_name=llm_model_name,
                    host="http://localhost:11434",
                    thinking=thinking,
                    timeout=300
                )

                if response is None:
                    print(f"[Step {step}] Resposta do LLM inválida. Pulando...")
                    continue

                raw_json = filter_generated_response(generated_sentence=response)

                if not raw_json:
                    print(f"[Step {step}] Nenhum JSON extraído. Pulando...")
                    continue

                parsed = json.loads(raw_json)
                config_llm = NASConfig.model_validate(parsed).model_dump()

                cfg_key = json.dumps(config_llm, sort_keys=True)

                if cfg_key in tested_configs:
                    print(f"[Step {step}] Config repetida. Pulando...")
                    continue

                tested_configs.add(cfg_key)

                print(f"[Step {step}] Config válida:")
                print(config_llm)

                # ================= MODEL TRAIN =================
                dynamic_model = dynamicMultimodalmodel.DynamicCNN(
                    config_llm,
                    num_classes=num_classes,
                    device=device,
                    common_dim=config_llm["common_dim"],
                    num_heads=num_heads,
                    vocab_size=num_metadata_features,
                    attention_mecanism=config_llm["attention_mechanism"],
                    n=1 if config_llm["attention_mechanism"] == "no-metadata" else 2
                )

                _, metrics = train_process(
                    config=config_llm,
                    num_epochs=num_epochs,
                    num_heads=num_heads,
                    fold_num=step,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    targets=val_dataset.targets,
                    model=dynamic_model,
                    device=device,
                    class_weights=class_weights,
                    common_dim=config_llm["common_dim"],
                    model_name=model_name,
                    text_model_encoder=text_model_encoder,
                    attention_mechanism=config_llm["attention_mechanism"],
                    llm_model_name=llm_model_name,
                    results_folder_path=results_folder_path
                )

                reward = float(metrics["balanced_accuracy"])
                val_loss = float(metrics["val_loss"])

                history.append({
                    "config": config_llm,
                    "reward": reward
                })

                with open(history_path, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "config": config_llm,
                        "reward": reward,
                        "val_loss": val_loss,
                        "metrics": metrics
                    }) + "\n")

                mlflow.log_metric("controller_reward", reward, step=step)
                mlflow.log_metric("controller_val_loss", val_loss, step=step)
                mlflow.log_param(f"config_step_{step}", json.dumps(config_llm))

                if reward > best_reward:
                    best_reward = reward
                    best_config = config_llm
                    best_step = step

                    with open(best_cfg_path, "w") as f:
                        json.dump({
                            "step": best_step,
                            "reward": best_reward,
                            "config": best_config
                        }, f, indent=2)

                    print(f"🏆 New best BACC = {best_reward:.4f}")

            except (json.JSONDecodeError, ValidationError) as e:
                print(f"[Step {step}] Config inválida descartada:")
                print(e)

            except RuntimeError as e:
                print(f"[Step {step}] RuntimeError durante treino:")
                print(e)

            except Exception as e:
                print(f"[Step {step}] Erro inesperado:")
                print(e)

            finally:
                if dynamic_model is not None:
                    try:
                        dynamic_model.to("cpu")
                    except Exception:
                        pass

                dynamic_model = None
                metrics = None
                config_llm = None
                reward = None
                val_loss = None

                cleanup_cuda()

        print("\n--- NAS FINALIZADO ---")
        print(f"Melhor BACC: {best_reward:.4f}")
        print(f"Step: {best_step}")
        print(f"Arquitetura: {best_config}")

        mlflow.log_metric(
            "final_best_reward",
            best_reward if best_reward != -float("inf") else 0.0
        )

        if best_config is not None:
            mlflow.log_param(
                "final_best_architecture_config",
                json.dumps(best_config)
            )

        print(f"Histórico salvo em: {history_path}")
        print(f"Best config salva em: {best_cfg_path}")

if __name__ == "__main__":
    # Carrega os dados localmente
    local_variables = load_local_variables.get_env_variables()
    num_epochs = local_variables["num_epochs"]  # Treino com poucas épocas
    batch_size = local_variables["batch_size"]
    k_folds = 1  # Treino com poucas épocas
    common_dim = -1
    list_num_heads = local_variables["list_num_heads"]
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = local_variables["dataset_folder_path"]
    unfreeze_weights = str(local_variables["unfreeze_weights"])
    llm_model_name_sequence_generator = local_variables["LLM_MODEL_NAME_SEQUENCE_GENERATOR"]
    results_folder_path = local_variables["results_folder_path"]
    HISTORY_MODE=local_variables["HISTORY_MODE"]
    thinking=False  # Desabilita o "thinking" para evitar que o LLM gere explicações longas e se concentre apenas na configuração. Isso é importante para manter a resposta dentro do formato JSON esperado e evitar erros de parsing.
    results_folder_path = f"{results_folder_path}/{HISTORY_MODE}/{'with-thinking' if thinking else 'no-thinking'}/controller-{llm_model_name_sequence_generator}/{dataset_folder_name}/{'unfrozen_weights' if unfreeze_weights else 'frozen_weights'}"
    SEARCH_STEPS=local_variables["SEARCH_STEPS"]
    # Métricas para o experimento
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model_encoder = 'one-hot-encoder'  # "tab-transformer" # 'bert-base-uncased' # 'gpt2'
    num_workers=int(local_variables["num_workers"])
    # Para todas os tipos de estratégias a serem usadas
    list_of_attention_mecanism = ["custom-attention-mechanism"]
    # Testar com todos os modelos
    list_of_models = ["custom-cnn-with-NAS"]
    dataset = skinLesionDatasetsPAD2020.SkinLesionDataset(
        metadata_file=f"{dataset_folder_path}/metadata.csv",
        img_dir=f"{dataset_folder_path}/images",
        image_encoder="custom-cnn-with-NAS",
        bert_model_name="one-hot-encoder",
        drop_nan=False,
        size=(224, 224)
    )

    search_space = {
        "num_blocks": [2, 5, 10],
        "initial_filters": [16, 32, 64],
        "kernel_size": [3, 5],
        "layers_per_block": [1, 2],
        "use_pooling": [True, False],
        "common_dim": [64, 128, 256, 512],
        "attention_mechanism": ["no-metadata", "concatenation", "crossattention", "metablock"],
        "num_layers_text_fc": [1, 2, 3],
        "neurons_per_layer_size_of_text_fc": [64, 128, 256, 512],
        "num_layers_fc_module": [1, 2],
        "neurons_per_layer_size_of_fc_module": [256, 512]
    }

    pipeline(
        dataset=dataset,
        num_metadata_features=dataset.features.shape[1],
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        num_classes=len(np.unique(dataset.labels)),
        model_name="custom-cnn-with-NAS",
        num_heads=list_num_heads[0],
        text_model_encoder="one-hot-encoder",
        attention_mechanism=list_of_attention_mecanism[0],
        results_folder_path=results_folder_path,
        num_workers=num_workers,
        SEARCH_STEPS=int(SEARCH_STEPS),
        search_space=search_space,
        history_mode=HISTORY_MODE, 
        history_k=10,
        thinking=thinking,
        llm_model_name=llm_model_name_sequence_generator
    )
