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
    def __init__(
        self,
        metadata_file: str,
        img_dir: str,
        bert_model_name="one-hot-encoder",
        size: tuple = (224, 224),
        drop_nan: bool = False,
        random_undersampling: bool = False,
        image_encoder: str = "resnet-50",
        is_train: bool = True,
        type_of_problem: str = "multiclass"
    ):
        self.metadata_file = metadata_file
        self.img_dir = img_dir
        self.size = size
        self.bert_model_name = bert_model_name
        self.is_to_drop_nan = drop_nan
        self.random_undersampling = random_undersampling
        self.image_encoder = image_encoder
        self.is_train = is_train
        self.type_of_problem = type_of_problem.lower().strip()

        if self.type_of_problem not in {"binaryclass", "multiclass"}:
            raise ValueError(
                f"type_of_problem deve ser 'binaryclass' ou 'multiclass'. "
                f"Recebido: {type_of_problem}"
            )

        self.targets = None
        self.normalization = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = self.load_transforms()

        self.metadata = self.load_metadata()
        self.features, self.labels, self.targets = self.one_hot_encoding()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_name = self.metadata.iloc[idx]["img_id"]
        img_path = os.path.abspath(os.path.join(self.img_dir, image_name))

        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                image = np.array(image)
        except Exception as e:
            print(f"[Erro] Não foi possível abrir imagem com PIL: {img_path} — {e}")
            raise FileNotFoundError(f"Imagem inválida: {img_path}")

        if self.transform:
            image = self.transform(image=image)["image"]

        metadata = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)

        return image_name, image, metadata, label

    def load_transforms(self):
        if self.is_train:
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),
                A.Rotate(
                    limit=45,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.GaussianBlur(sigma_limit=(0, 2.0), p=0.25),
                A.CoarseDropout(
                    max_holes=5,
                    max_height=8,
                    max_width=8,
                    p=0.15
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=15,
                    val_shift_limit=10,
                    p=0.25
                ),
                A.RandomBrightnessContrast(p=0.25),
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.size[0], self.size[1]),
                A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
                ToTensorV2(),
            ])

    def load_metadata(self):
        metadata = (
            pd.read_csv(self.metadata_file)
            .fillna("EMPTY")
            .replace(" ", "EMPTY")
            .replace("  ", "EMPTY")
            .replace("NÃO  ENCONTRADO", "EMPTY")
            .replace("BRASIL", "BRAZIL")
        )

        if self.is_to_drop_nan:
            metadata = metadata.dropna().reset_index(drop=True)

        required_columns = [
            "patient_id", "lesion_id", "img_id", "biopsed", "diagnostic",
            "age", "diameter_1", "diameter_2"
        ]
        missing_cols = [col for col in required_columns if col not in metadata.columns]
        if missing_cols:
            raise ValueError(f"Colunas obrigatórias ausentes no metadata: {missing_cols}")

        return metadata.reset_index(drop=True)

    def _build_binary_target_column(self):
        benign_categories = {"ACK", "NEV", "SEK"}
        malignant_categories = {"BCC", "SCC", "MEL"}

        self.metadata["benign_malignant"] = self.metadata["diagnostic"].apply(
            lambda x: (
                "benign" if x in benign_categories
                else "malignant" if x in malignant_categories
                else "unknown"
            )
        )

        unknown_count = int((self.metadata["benign_malignant"] == "unknown").sum())
        if unknown_count > 0:
            print(
                f"[Aviso] {unknown_count} amostras ficaram com classe 'unknown' "
                f"na conversão para binaryclass."
            )

    def one_hot_encoding(self):
        metadata = self.metadata.copy()

        if self.type_of_problem == "binaryclass":
            self._build_binary_target_column()
            metadata = self.metadata.copy()
            target_column = "benign_malignant"
            encoder_suffix = "_binaryclass"
        else:
            target_column = "diagnostic"
            encoder_suffix = "_multiclass"

        dataset_features = metadata.drop(
            columns=["patient_id", "lesion_id", "img_id", "biopsed", "diagnostic"],
            errors="ignore"
        )

        if self.type_of_problem == "binaryclass":
            dataset_features = dataset_features.drop(columns=["benign_malignant"], errors="ignore")

        numerical_cols = ["age", "diameter_1", "diameter_2"]
        categorical_cols = [col for col in dataset_features.columns if col not in numerical_cols]

        dataset_features = dataset_features.copy()

        if categorical_cols:
            dataset_features[categorical_cols] = dataset_features[categorical_cols].astype(str)

        dataset_features[numerical_cols] = dataset_features[numerical_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        dataset_features[numerical_cols] = dataset_features[numerical_cols].fillna(-1)

        base_dir = os.path.join("./data", "preprocess_data")
        os.makedirs(base_dir, exist_ok=True)

        ohe_path = os.path.join(base_dir, f"ohe_pad_20{encoder_suffix}.pickle")
        scaler_path = os.path.join(base_dir, f"scaler_pad_20{encoder_suffix}.pickle")
        label_encoder_path = os.path.join(base_dir, f"label_encoder_pad_20{encoder_suffix}.pickle")

        if categorical_cols:
            if os.path.exists(ohe_path):
                with open(ohe_path, "rb") as f:
                    ohe = pickle.load(f)
                categorical_data = ohe.transform(dataset_features[categorical_cols])
            else:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                categorical_data = ohe.fit_transform(dataset_features[categorical_cols])
                with open(ohe_path, "wb") as f:
                    pickle.dump(ohe, f)
        else:
            categorical_data = np.empty((len(dataset_features), 0), dtype=np.float32)

        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            numerical_data = scaler.transform(dataset_features[numerical_cols])
        else:
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(dataset_features[numerical_cols])
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

        processed_data = np.hstack((categorical_data, numerical_data)).astype(np.float32)

        labels = metadata[target_column].astype(str).values

        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
            encoded_labels = label_encoder.transform(labels)
        else:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            with open(label_encoder_path, "wb") as f:
                pickle.dump(label_encoder, f)

        targets = np.array(label_encoder.classes_)

        print(f"[INFO] type_of_problem: {self.type_of_problem}")
        print(f"[INFO] target_column: {target_column}")
        print(f"[INFO] targets (label_encoder.classes_): {targets}")
        print(f"[INFO] labels únicos codificados: {np.unique(encoded_labels)}")
        print(f"[INFO] número de classes: {len(targets)}")
        print(f"[INFO] shape features: {processed_data.shape}")

        return processed_data, encoded_labels, targets