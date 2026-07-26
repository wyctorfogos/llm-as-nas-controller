from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
import pickle
import cv2

class SkinLesionDataset(Dataset):
    def __init__(self, metadata_file:str, img_dir:str, bert_model_name="one-hot-encoder", size:tuple=(224,224),
        drop_nan:bool=False, random_undersampling:bool=False,
        image_encoder:str="resnet-50", is_train:bool=True):
        # Store parameters
        self.metadata_file = metadata_file
        self.img_dir = img_dir
        self.size = size
        self.bert_model_name=bert_model_name
        self.is_to_drop_nan = drop_nan
        self.random_undersampling = random_undersampling
        self.image_encoder = image_encoder
        self.is_train = is_train
        self.targets = None
        self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = self.load_transforms()

        # Load metadata and process
        self.metadata = self.load_metadata()
        self.features, self.labels, self.targets = self.one_hot_encoding()


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_name = self.metadata.iloc[idx]['img_id']
        img_path = os.path.abspath(os.path.join(self.img_dir, image_name))

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = np.array(image)
        except Exception as e:
            print(f"[Erro] Não foi possível abrir imagem com PIL: {img_path} — {e}")
            raise FileNotFoundError(f"Imagem inválida: {img_path}")

        if self.transform:
            image = self.transform(image=image)['image']

        metadata = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_name, image, metadata, label

    def load_transforms(self):
        """
        Define as transformações de imagem para treino/validação.

        - Treino:
            * Resize fixo
            * Rotate moderado (±45°) – implementação própria do Albumentations (cv2), sem skimage.AffineTransform
            * Flips horizontal/vertical
            * Blur, dropout e variação de cor
            * Normalização + ToTensorV2

        - Val/Test:
            * Apenas Resize + Normalize + ToTensorV2
        """
        if self.is_train:
            return A.Compose([
                # Ajuste de tamanho base
                A.Resize(self.size[0], self.size[1]),

                # Geométricas SEGURAS (sem Affine / ShiftScaleRotate)
                A.Rotate(
                    limit=45,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5
                ),

                # Flips
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),

                # Blur
                A.GaussianBlur(sigma_limit=(0, 2.0), p=0.25),

                # Dropout leve (oclusões pequenas)
                A.CoarseDropout(
                    max_holes=5,
                    max_height=8,
                    max_width=8,
                    p=0.15
                ),

                # Variações de cor/iluminação
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.25
                ),
                A.RandomBrightnessContrast(p=0.25),

                # Normalização + tensor
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2(),
            ])
        else:
            # Validação / teste: sem augmentations fortes
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2(),
            ])


    def load_metadata(self):
        # Carregar o CSV
        metadata = pd.read_csv(self.metadata_file).fillna("EMPTY").replace(" ", "EMPTY").replace("  ", "EMPTY").\
           replace("NÃO  ENCONTRADO", "EMPTY").replace("BRASIL","BRAZIL")
        # Verificar se deve descartar linhas com NaN
        if self.is_to_drop_nan:
            metadata = metadata.dropna().reset_index(drop=True)

        return metadata

    def one_hot_encoding(self):
        dataset_features = self.metadata.drop(
            columns=['patient_id', 'lesion_id', 'img_id', 'biopsed', 'diagnostic']
        )

        # Colunas numéricas fixas
        numerical_cols = ['age', 'diameter_1', 'diameter_2']
        categorical_cols = [col for col in dataset_features.columns if col not in numerical_cols]

        # Converter categóricas
        dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)

        # Forçar numérico nas colunas numéricas, substituindo inválidos por NaN
        dataset_features[numerical_cols] = dataset_features[numerical_cols].apply(
            pd.to_numeric, errors="coerce"
        )

        # Preencher valores faltantes (NaN gerados acima) com -1
        dataset_features[numerical_cols] = dataset_features[numerical_cols].fillna(-1)


        # Caminho base
        base_dir = os.path.join("./data", "preprocess_data")
        os.makedirs(base_dir, exist_ok=True)

        # OneHotEncoder
        ohe_path = os.path.join(base_dir, "ohe_pad_20.pickle")
        if os.path.exists(ohe_path):
            with open(ohe_path, "rb") as f:
                ohe = pickle.load(f)
            categorical_data = ohe.transform(dataset_features[categorical_cols])
        else:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            categorical_data = ohe.fit_transform(dataset_features[categorical_cols])
            with open(ohe_path, "wb") as f:
                pickle.dump(ohe, f)

        # StandardScaler
        scaler_path = os.path.join(base_dir, "scaler_pad_20.pickle")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            numerical_data = scaler.transform(dataset_features[numerical_cols])
        else:
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(dataset_features[numerical_cols])
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        # Concatenar dados
        processed_data = np.hstack((categorical_data, numerical_data))

        # Labels
        labels = self.metadata['diagnostic'].values
        le_path = os.path.join(base_dir, "label_encoder_pad_20.pickle")
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                label_encoder = pickle.load(f)
            encoded_labels = label_encoder.transform(labels)
        else:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            with open(le_path, "wb") as f:
                pickle.dump(label_encoder, f)

        return processed_data, encoded_labels, self.metadata['diagnostic'].unique()
