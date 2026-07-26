import torch
import torch.nn as nn
import os
import gc
import random
import csv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models import skinLesionDatasetsPAD2020
from utils import model_metrics, save_predictions
from utils.early_stopping import EarlyStopping
from utils import load_local_variables
from models import dynamicMultimodalmodel
from models import skinLesionDatasetsWithBert
from utils.save_model_and_metrics import save_model_and_metrics
from sklearn.model_selection import train_test_split
import time
from torch.utils.data import DataLoader
import numpy as np
import mlflow
from tqdm import tqdm
import json
import re
import logging


def cleanup_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def safe_json_parse(raw_response: str):
    try:
        cleaned = re.sub(
            r"^```(?:json)?",
            "",
            raw_response.strip(),
            flags=re.IGNORECASE | re.MULTILINE
        )
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)

        return json.loads(cleaned)

    except Exception as e:
        print(f"[safe_json_parse] Erro no parsing: {e}")
        print("Resposta bruta:", raw_response)
        return None


def parse_bool(value):
    """
    Converts environment values to boolean safely.
    Useful when local_variables returns strings such as 'True' or 'False'.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in ["true", "1", "yes", "y"]

    return bool(value)


def compute_class_weights(labels, num_classes):
    """
    Computes class weights for CrossEntropyLoss, ensuring one weight per class.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)

    weights = []
    for i in range(num_classes):
        if counts[i] > 0:
            weight = total_samples / (num_classes * counts[i])
        else:
            weight = 0.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)


def sample_random_config(search_space: dict) -> dict:
    """
    Randomly samples one valid architecture configuration from the search space.
    This is used as a non-LLM NAS baseline under the same search budget.
    """
    return {
        key: random.choice(values)
        for key, values in search_space.items()
    }


def sample_unique_random_config(search_space: dict, sampled_configs: set, max_attempts: int = 1000):
    """
    Samples a random architecture without repetition whenever possible.
    """
    for _ in range(max_attempts):
        config = sample_random_config(search_space)
        config_signature = json.dumps(config, sort_keys=True)

        if config_signature not in sampled_configs:
            sampled_configs.add(config_signature)
            return config

    return None


def train_process(
        config: dict,
        num_epochs: int,
        num_heads: int,
        fold_num: int,
        train_loader,
        val_loader,
        targets,
        model,
        device: str,
        weightes_per_category,
        common_dim,
        model_name: str,
        text_model_encoder,
        attention_mecanism: str,
        results_folder_path: str
    ):
    try:
            
        criterion = nn.CrossEntropyLoss(weight=weightes_per_category)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=5e-5,
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=2,
            verbose=True
        )

        model.to(device)

        model_save_path = os.path.join(
            results_folder_path,
            f"model_{model_name}_with_{text_model_encoder}_{common_dim}_with_best_architecture"
        )

        os.makedirs(model_save_path, exist_ok=True)
        print(model_save_path)

        early_stopping = EarlyStopping(
            patience=10,
            delta=0.01,
            verbose=True,
            path=str(model_save_path + f'/step_{str(fold_num)}/best-model/'),
            save_to_disk=False,
            early_stopping_metric_name="val_bacc"
        )

        initial_time = time.time()
        epoch_index = 0
        train_losses, val_losses = [], []

        experiment_name = f"EXPERIMENTOS-NAS-{dataset_folder_name} -- RANDOM SEARCH"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=(
                f"image_extractor_model_{model_name}_with_mecanism_"
                f"{attention_mecanism}_step_{fold_num}_num_heads_{num_heads}"
            ),
            nested=True
        ):
            mlflow.log_param("step", fold_num)
            mlflow.log_param("batch_size", train_loader.batch_size)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("attention_mecanism", attention_mecanism)
            mlflow.log_param("text_model_encoder", text_model_encoder)
            mlflow.log_param("criterion_type", "cross_entropy")
            mlflow.log_param("num_heads", num_heads)
            mlflow.log_param("random_search_config", json.dumps(config))

            for epoch_index in range(num_epochs):
                model.train()
                running_loss = 0.0

                for batch_index, (_, image, metadata, label) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch_index + 1}/{num_epochs}", leave=False)
                ):
                    image = image.to(device)
                    metadata = metadata.to(device)
                    label = label.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                train_loss = running_loss / len(train_loader)
                train_losses.append(float(train_loss))

                logging.info(f"\nTraining: Epoch {epoch_index}, Loss: {train_loss:.4f}")

                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for _, image, metadata, label in val_loader:
                        image = image.to(device)
                        metadata = metadata.to(device)
                        label = label.to(device)

                        outputs = model(image, metadata)
                        loss = criterion(outputs, label)
                        val_loss += loss.item()

                val_loss = val_loss / len(val_loader)
                val_losses.append(float(val_loss))

                print(f"Validation Loss: {val_loss:.4f}")

                scheduler.step(val_loss)

                current_lr = [pg['lr'] for pg in optimizer.param_groups]
                print(f"Current Learning Rate(s): {current_lr}\n")

                metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    fold_num=fold_num,
                    targets=targets,
                    base_dir=model_save_path,
                    model_name=model_name
                )

                metrics["epoch"] = epoch_index
                metrics["train_loss"] = float(train_loss)
                metrics["val_loss"] = float(val_loss)
                metrics["attention_mechanism"] = str(attention_mecanism)
                metrics["common_dim"] = int(common_dim)

                logging.info(f"Metrics: {metrics}\n")

                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value, step=epoch_index + 1)
                    else:
                        mlflow.log_param(metric_name, metric_value)

                early_stopping(
                    val_loss=val_loss,
                    val_bacc=float(metrics["balanced_accuracy"]),
                    model=model
                )

                if early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break

        train_process_time = time.time() - initial_time

        model = early_stopping.load_best_weights(model)
        model.eval()

        with torch.no_grad():
            metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                fold_num=fold_num,
                targets=targets,
                base_dir=model_save_path,
                model_name=model_name
            )

        metrics["train process time"] = str(train_process_time)
        metrics["epochs"] = str(int(epoch_index))
        metrics["data_val"] = "val"
        metrics["epoch"] = epoch_index
        metrics["train_loss"] = float(train_loss)
        metrics["val_loss"] = float(val_loss)
        metrics["attention_mechanism"] = str(attention_mecanism)
        metrics["common_dim"] = int(common_dim)

        logging.info(f"Model saved at {model_save_path}")

        folder_name = f"{model_name}_fold_{fold_num}"
        folder_path = os.path.join(model_save_path, folder_name)
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

        metrics_file = os.path.join(results_folder_path, "all_model_metrics.csv")
        file_exists = os.path.isfile(metrics_file)

        with open(metrics_file, mode='a', newline='') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=metrics.keys(),
                extrasaction="ignore"
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics)

        with open(os.path.join(folder_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        del optimizer
        del scheduler
        del criterion
        cleanup_cuda()

        return model, model_save_path, metrics
    
    except Exception as e:
        logging.error(f"Erro no treino do modelo: {e}")
        raise ValueError(f"Erro no treino do modelo: {e}")

def load_random_search_checkpoint(history_path: str):
    history = []
    sampled_configs = set()

    best_reward = -float("inf")
    best_config = None
    best_step = -1
    last_step = 0

    if not os.path.isfile(history_path):
        return history, sampled_configs, best_reward, best_config, best_step, last_step

    with open(history_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            step = int(record.get("step", 0))
            config = record.get("config")
            reward = float(record.get("reward", 0.0))

            if config is None:
                continue

            config_signature = json.dumps(config, sort_keys=True)
            sampled_configs.add(config_signature)

            history.append({
                "step": step,
                "config": config,
                "reward": reward
            })

            if step > last_step:
                last_step = step

            if reward > best_reward:
                best_reward = reward
                best_config = config
                best_step = step

    return history, sampled_configs, best_reward, best_config, best_step, last_step

def pipeline(
        dataset,
        num_metadata_features,
        num_epochs,
        batch_size,
        device,
        num_classes,
        model_name,
        num_heads,
        common_dim,
        k_folds,
        text_model_encoder,
        unfreeze_weights,
        attention_mecanism,
        results_folder_path,
        SEARCH_STEPS,
        search_space,
        num_workers=5,
        persistent_workers=False,
        test_size=0.2,
        controller_lr=1e-3,
        entropy_beta=0.01,
        grad_clip_norm=1.0
    ):

    os.makedirs(results_folder_path, exist_ok=True)

    labels = [dataset.labels[i] for i in range(len(dataset))]

    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=labels,
        random_state=42
    )

    logging.info("Usando split simples estratificado train/val para exploração NAS.")

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
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )

    train_labels = [labels[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels, num_classes).to(device)

    logging.info(f"Pesos das classes: {class_weights}")

    history_path = os.path.join(results_folder_path, "random_search_history.jsonl")

    history, sampled_configs, best_reward, best_config, best_step, last_step = (
        load_random_search_checkpoint(history_path)
    )

    logging.info(f"Histórico carregado de: {history_path}")
    logging.info(f"Arquiteturas já testadas: {len(sampled_configs)}")
    logging.info(f"Último step registrado: {last_step}")
    logging.info(f"Melhor reward anterior: {best_reward}")
    logging.info(f"Melhor step anterior: {best_step}")

    with mlflow.start_run(nested=True):
        mlflow.log_param("search_strategy", "random_search")
        mlflow.log_param("controller_learning_rate", controller_lr)
        mlflow.log_param("entropy_beta", entropy_beta)
        mlflow.log_param("gradient_clip_norm", grad_clip_norm)
        mlflow.log_param("search_steps", SEARCH_STEPS)
        mlflow.log_param("search_space_json", json.dumps(search_space))
        mlflow.log_param("train_val_split", "stratified_train_test_split")
        mlflow.log_param("test_size", test_size)

        target_total_steps = SEARCH_STEPS

        if last_step >= SEARCH_STEPS:
            logging.info(
                f"Busca já concluída: last_step={last_step}, SEARCH_STEPS={SEARCH_STEPS}"
            )
            return

        for step in range(last_step + 1, target_total_steps + 1):

            try:
                config_llm = sample_unique_random_config(
                    search_space=search_space,
                    sampled_configs=sampled_configs
                )

                if config_llm is None:
                    logging.info(f"[Step {step}] Não foi possível amostrar uma configuração nova.")
                    continue

                selected_common_dim = int(config_llm["common_dim"])
                selected_attention_mecanism = str(config_llm["attention_mecanism"])

                logging.info(f"[Random Search] Config amostrada: {config_llm}")

            except Exception as e:
                logging.exception(f"[Step {step}] Erro ao amostrar configuração aleatória: {e}")
                continue

            try:
                cleanup_cuda()

                dynamic_model = dynamicMultimodalmodel.DynamicCNN(
                    config=config_llm,
                    num_classes=num_classes,
                    device=device,
                    common_dim=selected_common_dim,
                    num_heads=num_heads,
                    vocab_size=num_metadata_features,
                    attention_mecanism=selected_attention_mecanism,
                    n=1 if selected_attention_mecanism == "no-metadata" else 2
                )

                dynamic_model, model_save_path, metrics = train_process(
                    config=config_llm,
                    num_epochs=num_epochs,
                    num_heads=num_heads,
                    fold_num=step,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    targets=dataset.targets,
                    model=dynamic_model,
                    device=device,
                    weightes_per_category=class_weights,
                    common_dim=selected_common_dim,
                    model_name=model_name,
                    text_model_encoder=text_model_encoder,
                    attention_mecanism=selected_attention_mecanism,
                    results_folder_path=results_folder_path
                )

                reward = metrics["balanced_accuracy"]
                dynamic_cnn_val_loss = metrics["val_loss"]

            except torch.cuda.OutOfMemoryError as e:
                logging.exception(f"[Step {step}] CUDA OOM com config {config_llm}: {e}")
                reward, dynamic_cnn_val_loss = 0.0, 1.0

            except Exception as e:
                logging.exception(f"Erro ao treinar modelo com config {config_llm}: {e}")
                reward, dynamic_cnn_val_loss = 0.0, 1.0

            finally:
                try:
                    if dynamic_model is not None:
                        dynamic_model.to("cpu")
                except Exception:
                    pass

                del dynamic_model
                del model_save_path
                del metrics

                cleanup_cuda()

            history.append({
                "step": step,
                "config": config_llm,
                "reward": reward
            })

            is_best_so_far = reward > best_reward

            history_record = {
                "step": step,
                "config": config_llm,
                "reward": float(reward),
                "val_loss": float(dynamic_cnn_val_loss),
                "is_best_so_far": bool(is_best_so_far),
                "best_reward_before_step": float(best_reward) if best_reward != -float('inf') else None
            }

            with open(os.path.join(results_folder_path, "random_search_history.jsonl"), "a") as f:
                f.write(json.dumps(history_record) + "\n")

            if is_best_so_far:
                best_reward = reward
                best_config = config_llm
                best_step = step

                with open(os.path.join(results_folder_path, "best_config.json"), "w") as f:
                    json.dump(best_config, f, indent=2)

            mlflow.log_metric("controller_reward", float(reward), step=step)
            mlflow.log_metric("dynamic-cnn-val_loss", float(dynamic_cnn_val_loss), step=step)
            # mlflow.log_param(f"config_step_{step}", json.dumps(config_llm))

            logging.info(
                f"[{step}/{SEARCH_STEPS}] "
                f"Reward: {reward:.4f} | "
                f"Best={best_reward:.4f} | "
                f"Config: {config_llm}"
            )

        logging.info("\n--- Busca Finalizada ---")
        logging.info(f"Melhor Reward: {best_reward:.4f}")
        logging.info(f"Melhor Arquitetura: {best_config}")
        logging.info(f"Step da melhor arquitetura: {best_step}")
        logging.info(f"Total de configurações únicas amostradas: {len(sampled_configs)}")

        mlflow.log_metric("final_best_reward", float(best_reward), step=SEARCH_STEPS)
        mlflow.log_param("final_best_architecture_config", json.dumps(best_config))
        mlflow.log_param("final_best_architecture_step", best_step)
        mlflow.log_param("total_unique_sampled_configs", len(sampled_configs))

        final_summary = {
            "search_strategy": "random_search",
            "search_steps": SEARCH_STEPS,
            "last_step": int(target_total_steps),
            "best_reward": float(best_reward),
            "best_step": int(best_step),
            "best_config": best_config,
            "total_unique_sampled_configs": len(sampled_configs),
            "resumed_from_step": int(last_step)
        }

        with open(os.path.join(results_folder_path, "random_search_summary.json"), "w") as f:
            json.dump(final_summary, f, indent=2)

        with open(os.path.join(results_folder_path, "best_config.json"), "w") as f:
            json.dump(best_config, f, indent=2)


def run_expirements(
        dataset_folder_path: str,
        results_folder_path: str,
        llm_model_name_sequence_generator: str,
        num_epochs: int,
        batch_size: int,
        k_folds: int,
        common_dim: int,
        text_model_encoder: str,
        unfreeze_weights,
        device,
        list_num_heads: list,
        list_of_attention_mecanism: list,
        list_of_models: list,
        SEARCH_STEPS,
        search_space
    ):

    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    if text_model_encoder in ['one-hot-encoder', "tab-transformer"]:
                        dataset = skinLesionDatasetsPAD2020.SkinLesionDataset(
                            metadata_file=f"{dataset_folder_path}/metadata.csv",
                            img_dir=f"{dataset_folder_path}/images",
                            bert_model_name=text_model_encoder,
                            image_encoder=model_name,
                            drop_nan=False,
                            size=(224, 224)
                        )

                    elif text_model_encoder in ['gpt2', 'bert-base-uncased']:
                        dataset = skinLesionDatasetsWithBert.SkinLesionDataset(
                            metadata_file=(
                                f"{dataset_folder_path}/"
                                f"metadata_with_sentences_new-prompt-{llm_model_name_sequence_generator}.csv"
                            ),
                            img_dir=f"{dataset_folder_path}/images",
                            bert_model_name=text_model_encoder,
                            image_encoder=model_name,
                            drop_nan=False,
                            size=(224, 224)
                        )

                    else:
                        raise ValueError("Encoder de texto não implementado!\n")

                    num_metadata_features = (
                        dataset.features.shape[1]
                        if text_model_encoder == 'one-hot-encoder'
                        else 512
                    )

                    logging.info(f"Número de features do metadados: {num_metadata_features}\n")

                    num_classes = len(dataset.metadata['diagnostic'].unique())

                    pipeline(
                        dataset=dataset,
                        num_metadata_features=num_metadata_features,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        device=device,
                        k_folds=-1,
                        num_classes=num_classes,
                        model_name=model_name,
                        common_dim=common_dim,
                        text_model_encoder=text_model_encoder,
                        num_heads=num_heads,
                        unfreeze_weights=unfreeze_weights,
                        attention_mecanism=attention_mecanism,
                        results_folder_path=f"{results_folder_path}/{num_heads}/{attention_mecanism}",
                        SEARCH_STEPS=SEARCH_STEPS,
                        search_space=search_space,
                        num_workers=6,
                        persistent_workers=False
                    )

                except Exception as e:
                    logging.exception(
                        f"Erro ao processar o treino do modelo {model_name} "
                        f"e com o mecanismo: {attention_mecanism}. Erro: {e}\n"
                    )
                    continue


if __name__ == "__main__":

    local_variables = load_local_variables.get_env_variables()

    num_epochs = local_variables["num_epochs"]
    batch_size = local_variables["batch_size"]
    k_folds = 1

    # This value is kept for compatibility, but Random Search uses config["common_dim"].
    common_dim = -1

    list_num_heads = local_variables["list_num_heads"]
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = local_variables["dataset_folder_path"]

    unfreeze_weights = parse_bool(local_variables["unfreeze_weights"])

    llm_model_name_sequence_generator = local_variables["LLM_MODEL_NAME_SEQUENCE_GENERATOR"]
    results_folder_path = local_variables["results_folder_path"]

    results_folder_path = (
        f"{results_folder_path}/"
        f"{dataset_folder_name}/"
        f"NAS/random_search/"
        f"{'unfrozen_weights' if unfreeze_weights else 'frozen_weights'}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    text_model_encoder = 'one-hot-encoder'

    # Kept for compatibility with the original script.
    # The actual fusion mechanism used during training is sampled from search_space["attention_mecanism"].
    list_of_attention_mecanism = ["random-search"]

    list_of_models = ["custom-cnn-with-NAS"]

    search_space = {
        "num_blocks": [2, 5, 10],
        "initial_filters": [16, 32, 64],
        "kernel_size": [3, 5],
        "layers_per_block": [1, 2],
        "use_pooling": [True, False],
        "common_dim": [64, 128, 256, 512],
        "attention_mecanism": [
            "no-metadata",
            "concatenation",
            "crossattention",
            "metablock"
        ],
        "num_layers_text_fc": [1, 2, 3],
        "neurons_per_layer_size_of_text_fc": [64, 128, 256, 512],
        "num_layers_fc_module": [1, 2],
        "neurons_per_layer_size_of_fc_module": [256, 512]
    }

    SEARCH_STEPS = 500

    logging.info(f"SEARCH_STEPS: {SEARCH_STEPS}\n")
    
    run_expirements(
        dataset_folder_path=dataset_folder_path,
        results_folder_path=results_folder_path,
        llm_model_name_sequence_generator=llm_model_name_sequence_generator,
        num_epochs=num_epochs,
        batch_size=batch_size,
        k_folds=k_folds,
        common_dim=common_dim,
        text_model_encoder=text_model_encoder,
        unfreeze_weights=unfreeze_weights,
        device=device,
        list_num_heads=list_num_heads,
        list_of_attention_mecanism=list_of_attention_mecanism,
        list_of_models=list_of_models,
        SEARCH_STEPS=SEARCH_STEPS,
        search_space=search_space
    )