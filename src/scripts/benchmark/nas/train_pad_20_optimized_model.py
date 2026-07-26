import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from models import skinLesionDatasetsPAD2020
from utils import model_metrics, load_local_variables, save_predictions, load_multimodal_config
from utils.early_stopping import EarlyStopping
from models import multimodalIntraInterModal, dynamicMultimodalmodel
from models import skinLesionDatasetsWithBert
from utils.save_model_and_metrics import save_model_and_metrics
from collections import Counter
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import json
import time
from torch.utils.data import DataLoader, Subset
import numpy as np
import mlflow
from tqdm import tqdm

# Função para calcular os pesos das classes garantindo que haja um peso para cada classe
def compute_class_weights(labels, num_classes):
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

def train_process(num_epochs, 
                  num_heads, 
                  fold_num, 
                  train_loader, 
                  val_loader, 
                  targets, 
                  model, 
                  device, 
                  weightes_per_category, 
                  common_dim, 
                  model_name, 
                  text_model_encoder, 
                  attention_mecanism, 
                  results_folder_path):

    criterion = nn.CrossEntropyLoss(weight=weightes_per_category)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
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
        path=str(model_save_path + f'/{model_name}_fold_{fold_num}/best-model/'),
        save_to_disk=True,
        early_stopping_metric_name="val_bacc"
    )

    initial_time = time.time()
    epoch_index = 0
    train_losses=[]
    val_losses=[]

    experiment_name = f"EXPERIMENTOS-{dataset_folder_name} - NAS WITH LLM AS CONTROLLER - TREINO DAS ARQUITETURAS ENCONTRADAS"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=(
            f"image_extractor_model_{model_name}_with_mecanism_"
            f"{attention_mecanism}_fold_{fold_num}_num_heads_{num_heads}"
        )
    ):
        mlflow.log_param("fold_num", fold_num)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("attention_mecanism", attention_mecanism)
        mlflow.log_param("text_model_encoder", text_model_encoder)
        # mlflow.log_param("criterion_type", "cross_entropy")
        mlflow.log_param("num_heads", num_heads)

        # Loop de treinamento
        for epoch_index in range(num_epochs):
            model.train()
            running_loss = 0.0

            for batch_index, ( _, image, metadata, label) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch_index+1}/{num_epochs}", leave=False)):
                image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                optimizer.zero_grad()
                outputs = model(image, metadata)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            print(f"\nTraining: Epoch {epoch_index}, Loss: {train_loss:.4f}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ , image, metadata, label in val_loader:
                    # print(f"Image names: {image_name}\n")
                    image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                    outputs = model(image, metadata)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {val_loss:.4f}")

            scheduler.step(val_loss)
            current_lr = [pg['lr'] for pg in optimizer.param_groups]
            print(f"Current Learning Rate(s): {current_lr}\n")

            metrics, all_labels, all_predictions, all_probs = model_metrics.evaluate_model(

                model=model, dataloader = val_loader, device=device, fold_num=fold_num, targets=targets, base_dir=model_save_path, model_name=model_name 
            )
            metrics["epoch"] = epoch_index
            metrics["train_loss"] = float(train_loss)
            metrics["val_loss"] = float(val_loss)
            print(f"Metrics: {metrics}")

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value, step=epoch_index + 1)
                else:
                    mlflow.log_param(metric_name, metric_value)

            early_stopping(val_loss=val_loss, val_bacc=float(metrics["balanced_accuracy"]), model=model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            # Salvar os pesos no array
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
    
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
    metrics["epochs"] = str(int(epoch_index))
    metrics["data_val"] = "val"

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
    print(f"Model saved at {model_save_path}")

    return model, model_save_path

def pipeline(dataset, num_metadata_features, num_epochs, batch_size, device, multimodel_config, type_of_problem, k_folds, num_classes, model_name, num_heads, common_dim, text_model_encoder, unfreeze_weights, attention_mecanism, results_folder_path, num_workers=10, persistent_workers=False):
    # Separação por paciente
    labels = dataset.labels                      # diagnóstico codificado
    groups = dataset.metadata["patient_id"].values  # agrupa por paciente
    stratifiedKFold = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(
        stratifiedKFold.split(X=np.zeros(len(labels)), y=labels, groups=groups)
    ):
        print(f"Fold {fold+1}/{k_folds}")
        
        # Validação: skip fold se não tiver pelo menos 2 classes na validação ou treino
        train_labels_fold = [labels[i] for i in train_idx]
        val_labels_fold = [labels[i] for i in val_idx]
        train_counts = Counter(train_labels_fold)
        val_counts = Counter(val_labels_fold)
        print(f"Fold {fold+1}: train={train_counts}, val={val_counts}")
        
        if len(val_counts) < 2:
            print(f"⚠️ Fold {fold+1} skipped: validation set has only {len(val_counts)} class(es). Classes: {set(val_labels_fold)}")
            continue
        if len(train_counts) < 2:
            print(f"⚠️ Fold {fold+1} skipped: training set has only {len(train_counts)} class(es). Classes: {set(train_labels_fold)}")
            continue
        
        # Verificar número mínimo de exemplares de cada classe
        MIN_SAMPLES_PER_CLASS = 5
        if min(val_counts.values()) < MIN_SAMPLES_PER_CLASS:
            print(f"⚠️ Fold {fold+1} skipped: validation has less than {MIN_SAMPLES_PER_CLASS} samples in some class. Counts: {val_counts}")
            continue
        if min(train_counts.values()) < MIN_SAMPLES_PER_CLASS:
            print(f"⚠️ Fold {fold+1} skipped: training has less than {MIN_SAMPLES_PER_CLASS} samples in some class. Counts: {train_counts}")
            continue

        train_dataset = type(dataset)(
            metadata_file=dataset.metadata_file,
            img_dir=dataset.img_dir,
            size=dataset.size,
            drop_nan=dataset.is_to_drop_nan,
            bert_model_name=dataset.bert_model_name,
            image_encoder=dataset.image_encoder,
            type_of_problem=type_of_problem,
            is_train=True  # Apply training augmentations
        )
        train_dataset.metadata = dataset.metadata.iloc[train_idx].reset_index(drop=True)
        train_dataset.features, train_dataset.labels, train_dataset.targets = train_dataset.one_hot_encoding()

        val_dataset = type(dataset)(
            metadata_file=dataset.metadata_file,
            img_dir=dataset.img_dir,
            size=dataset.size,
            drop_nan=dataset.is_to_drop_nan,
            bert_model_name=dataset.bert_model_name,
            image_encoder=dataset.image_encoder,
            type_of_problem=type_of_problem,
            is_train=False  # Apply validation transforms
        )
        val_dataset.metadata = dataset.metadata.iloc[val_idx].reset_index(drop=True)
        val_dataset.features, val_dataset.labels, val_dataset.targets = val_dataset.one_hot_encoding()

        # Criar DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)

        train_labels = [labels[i] for i in train_idx]
        class_weights = compute_class_weights(train_labels, num_classes).to(device)
        print(f"Pesos das classes no fold {fold+1}: {class_weights}")
        
        if (text_model_encoder in ["one-hot-encoder", "tab-transformer","gpt2", "bert-base-uncased"]):
            # Instancia o modelo dinâmico com a configuração amostrada
            model = dynamicMultimodalmodel.DynamicCNN(
                config=multimodel_config, num_classes=num_classes, device=device,
                    common_dim=common_dim, num_heads=num_heads, vocab_size=num_metadata_features,
                    attention_mecanism=attention_mecanism, 
                    n=1 if attention_mecanism == "no-metadata" else 2
                )

        else:
            raise ValueError("Encoder de texto não implementado!\n")

        # Treino do modelo carregado
        model, model_save_path = train_process(
            num_epochs, num_heads, fold+1, train_loader, val_loader, 
            dataset.targets, model, device, class_weights, 
            common_dim, model_name, text_model_encoder, attention_mecanism, results_folder_path
        )

        # Salvar as predições em um arquivo csv
        save_predictions.model_val_predictions(model=model, dataloader=val_loader, device=device, fold_num=fold+1,
            targets= dataset.targets, base_dir=model_save_path, model_name=model_name)    


def run_expirements(dataset_folder_path:str, results_folder_path:str, multimodel_config:dict, type_of_problem:str, llm_model_name_sequence_generator:str, num_epochs:int, batch_size:int, k_folds:int, common_dim:int, text_model_encoder:str, unfreeze_weights: str, device, list_num_heads: list, list_of_attention_mecanism:list, list_of_models: list):
    for attention_mecanism in list_of_attention_mecanism:
        for model_name in list_of_models:
            for num_heads in list_num_heads:
                try:
                    if (text_model_encoder in ['one-hot-encoder', "tab-transformer"]):
                        dataset = skinLesionDatasetsPAD2020.SkinLesionDataset(
                        metadata_file=f"{dataset_folder_path}/metadata.csv",
                        img_dir=f"{dataset_folder_path}/images",
                        bert_model_name=text_model_encoder,
                        image_encoder=model_name,
                        type_of_problem=type_of_problem,
                        drop_nan=False,
                        size=(224,224))
                    else:
                        raise ValueError("Encoder de texto não implementado!\n")
                    
                    num_metadata_features = dataset.features.shape[1] if text_model_encoder == 'one-hot-encoder' else 512
                    print(f"Número de features do metadados: {num_metadata_features}\n")
                    num_classes = len(dataset.targets)

                    pipeline(dataset=dataset, 
                        num_metadata_features=num_metadata_features,
                        multimodel_config=multimodel_config, 
                        num_epochs=num_epochs, batch_size=batch_size, 
                        device=device, k_folds=k_folds, num_classes=num_classes, 
                        model_name=model_name, common_dim=common_dim, 
                        text_model_encoder=text_model_encoder,
                        num_heads=num_heads,
                        unfreeze_weights=unfreeze_weights,
                        attention_mecanism=attention_mecanism, 
                        results_folder_path=f"{results_folder_path}/{num_heads}",
                        num_workers=5, 
                        persistent_workers=False,
                        type_of_problem=type_of_problem
                    )
                except Exception as e:
                    print(f"Erro ao processar o treino do modelo {model_name} e com o mecanismo: {attention_mecanism}. Erro:{e}\n")
                    continue

if __name__ == "__main__":
    # Carrega os dados localmente
    local_variables=load_local_variables.get_env_variables()
    num_epochs = local_variables["num_epochs"]
    batch_size = local_variables["batch_size"]
    k_folds = local_variables["k_folds"]
    common_dim = local_variables["common_dim"]
    list_num_heads = local_variables["list_num_heads"]
    dataset_folder_name = local_variables["dataset_folder_name"]
    dataset_folder_path = local_variables["dataset_folder_path"]
    unfreeze_weights = str(local_variables["unfreeze_weights"])
    llm_model_name_sequence_generator=local_variables["LLM_MODEL_NAME_SEQUENCE_GENERATOR"]
    results_folder_path = local_variables["results_folder_path"]
    type_of_problem="binaryclass"  # "binaryclass" or "multiclass"
    results_folder_path = f"{results_folder_path}/{dataset_folder_name}/{type_of_problem}/{'unfrozen_weights' if unfreeze_weights else 'frozen_weights'}"
    # Métricas para o experimento
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model_encoder = 'one-hot-encoder' # "tab-transformer" # 'bert-base-uncased' # 'gpt2' # 'one-hot-encoder'

    # Caminho de onde está o arquivo com as melhores configurações encontrada no processo de treino do NAS
    # best_model_parameters_file_folder_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/NAS-USING-RL-USING-REWARD-500-steps/PAD-UFES-20/unfrozen_weights/8/custom-attention-mechanism/best_config.json"
    # config = load_multimodal_config.load_multimodal_config(best_model_parameters_file_folder_path)
    ## configs = load_multimodal_config.load_multimodal_config("./data/PAD-UFES-20/NAS_OPTIMIZED_MODEL_ARCHITECTURES/BEST_ARCHITECTURES_FOUND_NAS.json")
    # configs = [{"num_blocks": 10, "initial_filters": 64, "kernel_size": 3, "layers_per_block": 2, "use_pooling": True, "common_dim": 512, "attention_mechanism": "metablock", "num_layers_text_fc": 3, "neurons_per_layer_size_of_text_fc": 512, "num_layers_fc_module": 2, "neurons_per_layer_size_of_fc_module": 512}]
    configs = [load_multimodal_config.load_multimodal_config(json_file_folder_path="./results/NAS/25042026/PAD-UFES-20/NAS/random_search/frozen_weights/8/random-search/best_config.json")]
    
    print(configs)
    for i, config in enumerate(configs):    
        # Testar com todos os modelos
        list_of_models = [f"nas_multimodal_model_random-search"] # Lista dos modelos a serem treinados (via ID)
        list_of_attention_mecanism = [config.get("attention_mecanism")] 
        # Treina todos modelos que podem ser usados no modelo multi-modal
        run_expirements(
            dataset_folder_path=dataset_folder_path, 
            results_folder_path=results_folder_path,
            multimodel_config=config,
            llm_model_name_sequence_generator=llm_model_name_sequence_generator, 
            num_epochs=num_epochs, 
            batch_size=batch_size, 
            k_folds=k_folds, 
            common_dim=common_dim, 
            text_model_encoder=text_model_encoder, 
            unfreeze_weights=unfreeze_weights, 
            type_of_problem=type_of_problem,
            device=device, 
            list_num_heads=list_num_heads, 
            list_of_attention_mecanism=list_of_attention_mecanism, 
            list_of_models=list_of_models
        )
