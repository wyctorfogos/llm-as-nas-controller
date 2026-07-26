import os
import pandas as pd

def get_dataset_content(dataset_path: str):
    dataset = pd.read_csv(dataset_path, sep=",")
    return dataset

if __name__ == "__main__":
    all_results = []

    # list_of_attention_mecanism = [
    #     "no-metadata",
    #     "att-intramodal",
    #     "rg-att",
    #     "att-intramodal+residual",
    #     "cross-attention-only",
    #     "residual+cross-attention-metadados",
    #     "att-intramodal+residual+cross-attention-metadados",
    #     "att-intramodal+residual+cross-attention-metadados+rg-att2fusefeatures"
    # ]
    list_of_attention_mecanism = [
        "att-intramodal+residual+cross-attention-metadados",
        "rg-att-cross-modal",
        "rg-att-literal-text-description"
    ]
    dataset_name = "PAD-UFES-20"
    num_heads = 8
    list_state_of_weights = ["unfrozen_weights"]
    common_sizes = [512]

    #base_folder_path = (
    # f"./src/results/testes-da-implementacao-final_2/11042026-WITH-LN--METHOD-CONFIG-COMPARISON/{dataset_name}")
    
    # base_folder_path=f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/results/NAS/25042026/PAD-UFES-20/multiclass/unfrozen_weights/8/model_nas_multimodal_model_random-search_with_one-hot-encoder_512_with_best_architecture"
    base_folder_path=f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/data/04072026/val_bacc/PAD-UFES-20/multiclass"
    
    list_of_models = [
        "caformer_b36.sail_in22k_ft_in1k"
    ]

    # Ordem final desejada
    ordered_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "auc",
        "train process time",
        "epochs",
        "attention_mecanism",
        "model_name",
        "common_size"
    ]

    # Colunas que devem virar mean ± std
    numeric_columns_target = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "precision",
        "recall",
        "auc",
        "train process time",
        "epochs"
    ]

    for status_of_weigths in list_state_of_weights:
        for common_size in common_sizes:
            for attention_mecanism in list_of_attention_mecanism:
                for model_name in list_of_models:
                    dataset_folder_path = (
                        f"{base_folder_path}/{status_of_weigths}/{num_heads}/"
                        f"{attention_mecanism}/"
                        f"model_{model_name}_with_one-hot-encoder_{common_size}_with_best_architecture"
                    )
                    dataset_path = os.path.join(dataset_folder_path, "model_metrics.csv")
                    # dataset_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/results/NAS/25042026/PAD-UFES-20/multiclass/unfrozen_weights/8/model_nas_multimodal_model_random-search_with_one-hot-encoder_512_with_best_architecture/model_metrics.csv"

                    try:
                        dataset = get_dataset_content(dataset_path=dataset_path)

                        # Mantém apenas as colunas que realmente existem no CSV
                        numeric_columns = [col for col in numeric_columns_target if col in dataset.columns]

                        if not numeric_columns:
                            print(f"Nenhuma métrica válida encontrada em: {dataset_path}")
                            continue

                        dataset_valid = dataset.copy()

                        dataset_valid[numeric_columns] = dataset_valid[numeric_columns].apply(
                            pd.to_numeric, errors="coerce"
                        )

                        mean_values = dataset_valid[numeric_columns].mean()
                        std_values = dataset_valid[numeric_columns].std()

                        result_dict = {}

                        # Preenche todas as colunas numéricas no formato mean ± std
                        for col in numeric_columns_target:
                            if col in numeric_columns:
                                result_dict[col] = f"{mean_values[col]:.4f} ± {std_values[col]:.4f}"
                            else:
                                result_dict[col] = ""

                        # Metadados finais
                        result_dict["attention_mecanism"] = attention_mecanism
                        result_dict["model_name"] = model_name
                        result_dict["common_size"] = common_size

                        result_df = pd.DataFrame([result_dict])
                        all_results.append(result_df)

                    except Exception as e:
                        print(f"Erro ao processar {dataset_path}. Erro: {e}\n")
                        continue
                
    if not all_results:
        raise ValueError("Nenhum resultado foi carregado. Verifique os caminhos e os arquivos CSV.")

    final_results_df = pd.concat(all_results, ignore_index=True)

    # Garante a ordem exata das colunas
    for col in ordered_columns:
        if col not in final_results_df.columns:
            final_results_df[col] = ""

    final_results_df = final_results_df[ordered_columns]

    output_path = os.path.join(base_folder_path, "all_metric_values.csv")
    final_results_df.to_csv(output_path, index=False)

    print(f"Todos os resultados foram salvos em {output_path}.")