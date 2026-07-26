import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

def main(csv_file_path, output_dir, use_legend, dataset_name):
    # Carregar CSV
    df = pd.read_csv(csv_file_path)

    # Extrair média do 'balanced_accuracy' (remove ± e pega o primeiro número)
    df['BACC_mean'] = df['balanced_accuracy'].apply(lambda x: float(re.match(r'([0-9.]+)', x).group(1)))

    # Criar um rótulo combinado para diferenciar os modelos com mecanismos de atenção
    df['Model+Attention'] = df['model_name'] + "\n(" + df['attention_mecanism'] + ")"

    # Ordenar para manter consistência visual
    df = df.sort_values(by=['common_size', 'Model+Attention'])

    # Configurações do seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 7))

    # Criar gráfico de barras
    if use_legend is True:
        ax = sns.barplot(
            data=df,
            x='common_size',
            y='BACC_mean',
            hue='Model+Attention',
            palette='tab20'
        )
    else:
        ax = sns.barplot(
            data=df,
            x='common_size',
            y='BACC_mean',
            # hue='Model+Attention',
            palette='light:b'
        )
    # Customizações
    plt.title("Balanced Accuracy (BACC) average value grouped by projected features' common size", fontsize=14)
    plt.ylabel("Balanced Accuracy (BACC)", fontsize=14)
    plt.xlabel("Common size of projected features", fontsize=14)
    plt.ylim(0, 1)
    if use_legend is True:
        plt.legend(title="Attention mechanism", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(f"{output_dir}/bacc_all_common_sizes_grouped_{dataset_name}.png", dpi=400, bbox_inches='tight')
    plt.show()

    print(f"Gráfico salvo como 'bacc_all_common_sizes_grouped_{dataset_name}.png'")


if __name__=="__main__":
    dataset_name = "PAD-UFES-20" # "ISIC-2019" 
    # Caminho para o arquivo CSV
    # csv_file_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/differents_dimension_of_projected_features/{dataset_name}/unfrozen_weights/8/all_metric_values.csv"
    # csv_file_path ="/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/differents_dimension_of_projected_features/ISIC-2019/unfrozen_weights/8/all_metric_values.csv"
    # csv_file_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/differents_dimension_of_projected_features/PAD-UFES-20/unfrozen_weights/8/all_metric_values.csv"
    # csv_file_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/differents_dimension_of_projected_features/{dataset_name}/unfrozen_weights/8/all_metric_values.csv"
    csv_file_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/01012026/different_features_with_dimension_size/PAD-UFES-20/unfrozen_weights/8/all_metric_values.csv"
    output_dir = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/01012026/different_features_with_dimension_size/PAD-UFES-20/bacc_plots"
    os.makedirs(output_dir, exist_ok=True)

    main(csv_file_path=csv_file_path, output_dir=output_dir, use_legend=False, dataset_name=dataset_name)