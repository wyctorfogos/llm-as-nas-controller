import os
import pandas as pd

def load_dataset(file_path):
    """Carrega um dataset a partir de um arquivo CSV."""
    try:
        return pd.read_csv(filepath_or_buffer=file_path, sep=",")
    except Exception as e:
        print(f"Erro ao ler o arquivo '{file_path}'. Erro: {e}")
        return None

def convert_binary_to_categorical(df):
    """Converte colunas binárias em uma única coluna categórica."""
    category_columns = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    
    # Verifica se todas as colunas existem no dataset
    missing_cols = [col for col in category_columns if col not in df.columns]
    if missing_cols:
        print(f"Erro: Colunas ausentes no dataset: {missing_cols}")
        return None
    
    # Criar a coluna categórica baseada nos valores binários
    df["category"] = df[category_columns].idxmax(axis=1)

    # Remover as colunas binárias
    df = df.drop(columns=category_columns)

    return df

def concatenate_isic_dataset(file_folder_path):
    """Concatena os datasets ISIC-2019 GroundTruth e Metadata."""
    try:
        ground_truth_path = os.path.join(file_folder_path, "ISIC_2019_Training_GroundTruth.csv")
        metadata_path = os.path.join(file_folder_path, "ISIC_2019_Training_Metadata.csv")

        # Carregar os datasets
        ground_truth = load_dataset(ground_truth_path)
        metadata = load_dataset(metadata_path)

        if ground_truth is None or metadata is None:
            print("Erro: Não foi possível carregar um ou mais arquivos.")
            return

        # Converter os dados binários para categóricos
        ground_truth = convert_binary_to_categorical(ground_truth)
        if ground_truth is None:
            print("Erro ao converter os dados binários em categóricos.")
            return

        # Verificar se a coluna de junção "image" existe
        if "image" not in ground_truth.columns or "image" not in metadata.columns:
            print("Erro: Coluna 'image' não encontrada em um dos datasets.")
            return

        # Mesclar os datasets com base na coluna "image"
        merged_data = pd.merge(ground_truth, metadata, on="image", how="inner")

        # Salvar o dataset final
        output_path = os.path.join(file_folder_path, "training_full_metadata.csv")
        merged_data.to_csv(output_path, index=False)

        print(f"Dataset concatenado salvo em: {output_path}")

    except Exception as e:
        print(f"Erro ao juntar os dados. Erro: {e}")

if __name__ == "__main__":
    file_folder_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/datasets/ISIC-2019"
    
    # Concatena e gera os dados
    concatenate_isic_dataset(file_folder_path)
