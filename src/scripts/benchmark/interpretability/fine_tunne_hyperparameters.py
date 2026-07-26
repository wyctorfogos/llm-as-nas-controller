import torch
import torch.nn as nn
import os
import sys
from benchmark.models import skinLesionDatasetsPAD2020
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../benchmark')))
from benchmark.models import multimodalIntraInterModal
from  benchmark.utils.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import optuna
from  benchmark.utils import model_metrics
from sklearn.model_selection import train_test_split
import benchmark.models.focalLoss as focalLoss
from benchmark.models import multimodalIntraInterModal, multimodalToOptimize
from benchmark.utils.save_model_and_metrics import save_model_and_metrics
from benchmark.utils.save_experiments_log_for_opt import save_experiment_log
from collections import Counter
import numpy as np
import time
import mlflow

# Função para calcular os pesos das classes
def compute_class_weights(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    return torch.tensor([class_weights[cls] for cls in sorted(class_counts.keys())], dtype=torch.float)

# Função de treino
def train_model(train_loader, val_loader, dataset, model, device, class_weights, num_epochs, params, fold, model_name, text_model_encoder, attention_mecanism, results_folder_path):

    model.to(device)
    # criterion = focalLoss.FocalLoss(alpha=class_weights, gamma=2, reduction='mean')

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )
    # EarlyStopping
    early_stopping = EarlyStopping(patience=5, delta=0.01)
    # Registro do tempo de treinamento
    initial_time = time.time()
    best_val_loss = float('inf')
    # Setando o novo experimento
    experiment_name = "EXPERIMENTOS-PAD-UFES20-MODEL-86-FEATURES-OF-METADATA-OPT-MODEL-ARCHITECTURE"
    mlflow.set_experiment(experiment_name)
    # MLflow Logging
    with mlflow.start_run(run_name=f"image_extractor_model_{model_name}_with_mecanism_{attention_mecanism}_fold_{fold}_text_fc_{params['text_fc_config']['hidden_sizes']}_text_fc_dropout_{params['text_fc_config']['dropout']}_num_heads_{params['num_heads']}_fc_fusion_hidden_sizes_{params['fc_fusion_config']['hidden_sizes']}_fc_fusion_dropout_{params['fc_fusion_config']['dropout']}", nested = True):
        # Log static parameters
        mlflow.log_params({
            "text_fc_hidden_sizes": params['text_fc_config']['hidden_sizes'],
            "text_fc_dropout": params['text_fc_config']['dropout'],
            "num_heads": params['num_heads'],
            "fc_fusion_hidden_sizes": params['fc_fusion_config']['hidden_sizes'],
            "fc_fusion_dropout": params['fc_fusion_config']['dropout'],
            "model_name": model_name,
            "attention_mecanism": attention_mecanism,
            "fold": fold
        })

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            for image, metadata, label in train_loader:
                image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for image, metadata, label in val_loader:
                    image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            scheduler.step(val_loss)

            # Current learning rate
            current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]

            # Salvar o menor val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Log training and validation losses as metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr
            }, step=epoch)

            # Log evaluation metrics
            metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(
model, val_loader, device, fold, model_name=model_name)
            # Logar métricas no MLflow
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=epoch)
                else:
                    mlflow.log_param(metric_name, metric_value)  



            model_save_path = os.path.join(
                results_folder_path,
                f"model_{model_name}_att_{attention_mecanism}_textfc_{params['text_fc_config']['hidden_sizes'][0]}_dp{params['text_fc_config']['dropout']:.2f}_heads{params['num_heads']}_fusionfc_{params['fc_fusion_config']['hidden_sizes'][0]}_dp{params['fc_fusion_config']['dropout']:.2f}.pth"
            )

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     save_model_and_metrics(model, metrics, model_name, model_save_path, fold, all_labels, all_predictions, dataset.targets, data_val="val")
            # Check early stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    # Fim do treinamento
    train_process_time = time.time() - initial_time

    # Load the best model weights
    early_stopping.load_best_weights(model)

    # Adição do tempo de treino nos registros
    metrics["train process time"]=str(train_process_time)
    model_save_path = os.path.join(
        results_folder_path,
        f"model_{model_name}_att_{attention_mecanism}_textfc_{params['text_fc_config']['hidden_sizes'][0]}_dp{params['text_fc_config']['dropout']:.2f}_heads{params['num_heads']}_fusionfc_{params['fc_fusion_config']['hidden_sizes'][0]}_dp{params['fc_fusion_config']['dropout']:.2f}.pth"
    )
    save_model_and_metrics(model, metrics, model_name, model_save_path, fold, all_labels, all_predictions, dataset.targets, data_val="val")
    mlflow.log_artifact(model_save_path)  # Save model path to MLflow

    # Supondo que 'metrics' seja um dicionário com as métricas finais do experimento
    save_experiment_log(f"{results_folder_path}/experiment_log.csv", params, metrics)

    return best_val_loss


# Função de objetivo para Optuna
def objective(trial):
    batch_size = 128
    max_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "densenet169"
    text_model_encoder = "one-hot-encoder"
    attention_mecanism = "crossattention"

    params = {
        'text_fc_config': {
            'hidden_sizes': trial.suggest_categorical('hidden_sizes', [[1024, 512], [512, 256], [1024, 512, 256], [2048, 1024, 512, 128]]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5)
        },
        'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16, 32, 64, 128, 256, 512]),
        'fc_fusion_config': {
            'hidden_sizes': trial.suggest_categorical('fc_hidden_sizes', [[1024, 512], [512, 256, 128], [1024, 512, 256, 128]]),
            'dropout': trial.suggest_float('fc_dropout', 0.1, 0.5)
        }
    }

    dataset = skinLesionDatasetsPAD2020.SkinLesionDataset(
        metadata_file="/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/data/metadata.csv",
        img_dir="/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/data/images",
        bert_model_name="one-hot-encoder",
        image_encoder="densenet169",
        drop_nan=False,
        random_undersampling=False
    )
    
    # Hold-Out Split
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Calcular pesos das classes com base no conjunto de treino
    train_labels = [dataset.labels[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels).to(device)

    model = multimodalToOptimize.MultimodalModel(
        num_classes=len(dataset.metadata['diagnostic'].unique()),
        device=device,
        cnn_model_name=model_name,
        text_model_name=text_model_encoder,
        vocab_size=dataset.features.shape[1],
        attention_mecanism=attention_mecanism,
        text_fc_config=params['text_fc_config'],
        num_heads=params['num_heads'],
        fc_fusion_config=params['fc_fusion_config']
    )

    val_loss = train_model(
        train_loader, val_loader, dataset, model, device, class_weights,
        num_epochs=max_epochs, params=params, fold=0,
        model_name=model_name, text_model_encoder=text_model_encoder,
        attention_mecanism=attention_mecanism, results_folder_path="/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/86_features_metadata/fine-tunning"
    )

    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
