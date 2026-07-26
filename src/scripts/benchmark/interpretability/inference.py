import os
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from benchmark.models import multimodalIntraInterModal # Ajuste conforme a estrutura do seu projeto



def one_hot_encoding(metadata, ohe_path = "./data/preprocess_data/ohe_pad_20.pickle", scaler_path = "./data/preprocess_data/scaler_pad_20.pickle"):
    # Remover colunas desnecessárias e selecionar as features
    dataset_features = metadata.drop(columns=['patient_id', 'lesion_id', 'img_id', 'biopsed', 'diagnostic'])

    # Definir as colunas categóricas e numéricas corretamente
    for col in ['age', 'diameter_1', 'diameter_2', 'fitspatrick']:
        dataset_features[col] = pd.to_numeric(dataset_features[col], errors='coerce')

    # Identify categorical and numerical columns
    categorical_cols = dataset_features.select_dtypes(include=['object', 'bool']).columns
    numerical_cols = dataset_features.select_dtypes(include=['float64', 'int64']).columns
    # Converter categóricas para string
    dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)

    # Preencher valores faltantes nas colunas numéricas com a média da coluna
    dataset_features[numerical_cols] = dataset_features[numerical_cols].fillna(-1)
    print(f"{dataset_features[numerical_cols]}\n")
    # Assegurar que as colunas categóricas usadas na inferência correspondem às usadas no treinamento
    dataset_features_categorical = dataset_features[categorical_cols]

    # OneHotEncoder
    if os.path.exists(ohe_path):
        with open(ohe_path, 'rb') as f:
            ohe = pickle.load(f)
        categorical_data = ohe.transform(dataset_features_categorical)
    else:
        raise FileNotFoundError("Arquivo OneHotEncoder não encontrado. Certifique-se de que o modelo foi treinado corretamente.")

    # StandardScaler para dados numéricos
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        numerical_data = scaler.transform(dataset_features[numerical_cols])
    else:
        raise FileNotFoundError("Arquivo StandardScaler não encontrado. Certifique-se de que o modelo foi treinado corretamente.")

    # Concatenar os dados processados
    processed_metadata = np.hstack((categorical_data, numerical_data))
    return processed_metadata

def load_transforms():
    size=(224, 224)
    normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return A.Compose([
        A.Resize(size[0], size[1]),
        A.Normalize(mean=normalization[0], std=normalization[1]),
        ToTensorV2()
    ])

def process_image(img):
    image = img.convert("RGB")
    image = np.array(image)
    transform = load_transforms()
    return transform(image=image)['image']



def load_multimodal_model(device, model_path, num_classes=6, num_heads=2, vocab_size=85, cnn_model_name="densenet169", text_model_name="one-hot-encoder", attention_mecanism="concatenation"):
    try:
        model = multimodalIntraInterModal.MultimodalModel(
            num_classes=num_classes,
            num_heads=num_heads,
            device=device,
            cnn_model_name=cnn_model_name,
            text_model_name=text_model_name,
            vocab_size=vocab_size,
            attention_mecanism=attention_mecanism
        )
        model.to(device)
        model.eval()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Carregamento bem sucedido do modelo!\n")
    except Exception as e:
        raise SystemError("Erro ao carregar o modelo")
    return model

def inference(processed_image, processed_metadata, device, model_path, num_classes=6, num_heads=2, vocab_size=85, cnn_model_name="densenet169", text_model_name="one-hot-encoder", attention_mecanism="concatenation"):
    model = load_multimodal_model(device, model_path, num_classes, num_heads, vocab_size, cnn_model_name, text_model_name, attention_mecanism)
    
    # Adiciona a dimensão de batch
    processed_image = processed_image.unsqueeze(0).to(device)
    processed_metadata = torch.tensor(processed_metadata, dtype=torch.float32).to(device)
    print(processed_metadata.shape)
    print(processed_image.shape)
    with torch.no_grad():
        outputs = model(processed_image, processed_metadata)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    return predictions, probabilities


def process_data(text, column_names):
    # Dividir o texto em campos separados por vírgula
    data = pd.DataFrame([text.split(',')], columns=column_names)
    # Substituir valores vazios ou indesejados por "EMPTY"
    data=data.fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").\
           replace("NÃO  ENCONTRADO", "EMPTY")
    data = data.replace(r'^\s*$', 'EMPTY', regex=True)  # Substituir strings vazias ou espaços
    data = data.replace(['NÃO ENCONTRADO', 'n/a', None], 'EMPTY')  # Substituir indicadores comuns de valores vazios
    data = data.replace("BRASIL", "BRAZIL")
    return data

def get_target(wanted_label):
    target_index=-1
    LABELS = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"] # PAD-UFES-20
    for i in LABELS:
        if wanted_label==LABELS[i]:
            target_index        
    return target_index


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/PAD-UFES-20/unfrozen_weights/8/att-intramodal+residual+cross-attention-metadados/model_davit_tiny.msft_in1k_with_one-hot-encoder_512_with_best_architecture/davit_tiny.msft_in1k_fold_3/best-model/best_model.pt"

    # Carregar imagem de teste
    image = Image.open("./data/PAD-UFES-20/images/PAT_771_1491_390.png")
    # Processar imagem
    processed_image = process_image(img=image)  # Ajuste conforme o modelo de codificação de imagem
    # Definir nomes das colunas
    column_names = [
        "patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age",
        "pesticide", "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
        "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2", "diagnostic", "itch",
        "grew", "hurt", "changed", "bleed", "elevation", "img_id", "biopsed"
    ]

    # Carregar dados de teste
    text = "PAT_771,1491,True,True,ITALY,ITALY,69,False,MALE,False,True,True,True,3.0,FACE,6.0,3.0,BCC,True,UNK,False,UNK,False,True,PAT_771_1491_390.png,True"  # "PAT_1516,1765,,,,,8,,,,,,,,ARM,,,NEV,False,False,False,False,False,False,PAT_1516_1765_530.png,False"
    metadata = process_data(text, column_names)

    # Processar metadados
    processed_metadata = one_hot_encoding(metadata, ohe_path="./data/preprocess_data/ohe_pad_20.pickle", scaler_path="./data/preprocess_data/scaler_pad_20.pickle")
    print(f"Processed_metadata:{processed_metadata}\n")

    # Realizar inferência
    predictions, probabilities = inference(processed_image=processed_image, processed_metadata=processed_metadata, device=device, model_path=model_path, num_classes=6, num_heads=8, vocab_size=85, cnn_model_name="davit_tiny.msft_in1k", text_model_name="one-hot-encoder", attention_mecanism="att-intramodal+residual+cross-attention-metadados")

    print(f"Predictions: {predictions}\n")
    print(f"Probabilities: {probabilities}\n")
