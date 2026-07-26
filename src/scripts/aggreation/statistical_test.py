import numpy as np
from stats import statistical_test
import os
import pandas as pd

def load_dataset(folder_path):  
    dataset = pd.read_csv(folder_path, sep=',')
    return dataset

def get_metric_values(file_folder_path, metric_name):
    list_of_values_wanted_metric = []
    try:
        if os.path.exists(file_folder_path):
            csv_file = os.path.join(file_folder_path, 'model_metrics.csv')
            if os.path.exists(csv_file):  # Verifica se o arquivo CSV existe
                data = load_dataset(csv_file)
                # Verifica se a coluna 'bacc' existe antes de acessá-la
                if metric_name in data.columns:
                    # Exclui as duas últimas linhas
                    list_of_values_wanted_metric.append(data[metric_name].iloc[:-1].values)
                else:
                    print(f"{metric_name} não encontrada no arquivo {csv_file}\n")
            else:
                print(f"Arquivo CSV não encontrado: {csv_file}\n")
        else:
            print(f"Pasta não encontrada: {file_folder_path}\n")
            
        return list_of_values_wanted_metric[0].ravel()
    except Exception as e:
        raise ValueError(f"Erro ao obter as métricas: {e}")
    
def save_statistics_tests(test_results:str, model_name: str, encoder_method:str, file_folder_path: str):
    '''
        Função para salvar os resultados dos testes estatísticos
    '''
    try:
        dataframe = pd.DataFrame(data=test_results)
        dataframe.to_csv(file_folder_path+f"statistics_tests_results_{model_name}_{encoder_method}.csv")
    except Exception as e:
        print(f"Erro salvar os dados. Erro: {e}\n")

if __name__ == "__main__":
    # Métricas dos experimentos
    wanted_metric_list = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'auc']
    list_all_models_metrics = []
    list_all_models_metrics_all_lists = []
    method_names = []
    num_heads = 8
    dataset_name = "PAD-UFES-20" # "ISIC-2019
    base_folder_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/25032026/{dataset_name}/unfrozen_weights/{num_heads}"
    # Verificar se a pasta existe
    if not os.path.exists(base_folder_path):
        raise ValueError("Pasta não encontrada/não existe!")
    # Verifica os dados referentes aos experimentos via mecanismos de atenção
    list_of_attention_mecanism = ["rg-att", "att-intramodal+residual", "att-intramodal+residual+cross-attention-metadados", "att-intramodal+residual+cross-attention-metadados+att-intramodal+residual", "gfcam", "cross-weights-after-crossattention", "crossattention", "concatenation", "no-metadata", "weighted", "metablock"]
    for attention_mecanism in list_of_attention_mecanism:
        # Testar com todos os modelos
        list_of_models = ["mobilenet-v2", "densenet169", "resnet-50", "davit_tiny.msft_in1k", "mvitv2_small.fb_in1k", "coat_lite_small.in1k", "caformer_b36.sail_in22k_ft_in1k", "beitv2_large_patch16_224.in1k_ft_in22k_in1k"]
        for model_name in list_of_models:
            for wanted_metric_metric_name in wanted_metric_list:
                file_folder_path = f"{base_folder_path}/{attention_mecanism}/model_{model_name}_with_one-hot-encoder_512_with_best_architecture"
                aux_metric_values = get_metric_values(file_folder_path, wanted_metric_metric_name)  

                list_all_models_metrics.extend(aux_metric_values)
        list_all_models_metrics_all_lists.append(list_all_models_metrics)
        list_all_models_metrics=[]
        # Limpar a memória antes de analisar os próximos dados
    list_all_models_metrics=[]

    out=statistical_test(data=list_all_models_metrics_all_lists, alg_names=list_of_attention_mecanism)
    # Salvar os resultados
    save_statistics_tests(test_results={out}, model_name=model_name, encoder_method="all_methods", file_folder_path=base_folder_path)