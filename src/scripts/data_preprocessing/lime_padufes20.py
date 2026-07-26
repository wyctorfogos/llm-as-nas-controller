import pandas as pd
import numpy as np
import shap
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from lime.lime_tabular import LimeTabularExplainer

from models.skinLesionDatasets import SkinLesionDataset


# ======================================================
# 1) LOAD METADATA + FEATURES (NORMALIZED)
# ======================================================
dataset = SkinLesionDataset(
    metadata_file="./data/PAD-UFES-20/metadata.csv",
    img_dir="./data/PAD-UFES-20/images",
    bert_model_name="one-hot-encoder",
    image_encoder="davit-v2",
    drop_nan=False,
    size=(224, 224),
    is_train=False
)

meta = dataset.metadata.copy()

# ======================================================
# 2) LOAD MODEL PROBABILITIES
# ======================================================
df = pd.read_csv("./src/results/pad_metadata_with_probs_fold_1.csv")

target_cols = ["prob_NEV","prob_BCC","prob_ACK","prob_SEK","prob_SCC","prob_MEL"]

# ======================================================
# 3) MERGE CORRETO VIA img_id
# ======================================================
merged = pd.merge(meta, df[["img_id"] + target_cols], on="img_id", how="inner")

mask = meta["img_id"].isin(merged["img_id"])
X = dataset.features[mask]
y = merged[target_cols].values

print("Merged:", merged.shape)
print("Final X:", X.shape)
print("Final y:", y.shape)

# ======================================================
# 4) SURROGATE MODEL (MESMO DO SHAP)
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

print("Surrogate R²:", model.score(X_test, y_test))

# ======================================================
# 5) RECONSTRUIR NOMES DAS FEATURES (OHE + NUMERICAS)
# ======================================================
with open("../data/preprocess_data/ohe.pickle", "rb") as f:
    ohe = pickle.load(f)

categorical_cols = ohe.feature_names_in_
ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols))

numeric_cols = ["age", "diameter_1", "diameter_2"]
feature_names = ohe_feature_names + numeric_cols

print("Total de features:", len(feature_names))

# ======================================================
# 6) LIME EXPLAINER
# ======================================================
explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    mode="regression",
    discretize_continuous=False
)

os.makedirs("./src/results/lime_explanations", exist_ok=True)
os.makedirs("./src/results/lime_explanations/csv", exist_ok=True)

# ======================================================
# 7) EXPLICAR UMA AMOSTRA POR VEZ
# ======================================================
sample_idx = 10   # escolha uma linha para inspecionar
x_instance = X_test[sample_idx]

pred = model.predict(x_instance.reshape(1, -1))[0]
print("\nPredicted probabilities (surrogate):")
for t, v in zip(target_cols, pred):
    print(f"{t}: {v:.4f}")

# ======================================================
# 8) FAZER EXPLICAÇÃO LIME PARA CADA CLASSE
# ======================================================
for class_idx, class_name in enumerate(target_cols):

    print(f"\n🔎 LIME explanation for class: {class_name}")

    exp = explainer.explain_instance(
        data_row=x_instance,
        predict_fn=lambda x: model.predict(x)[:, class_idx],
        num_features=15
    )

    # ---- PRINT ----
    explanation_list = exp.as_list()
    print(explanation_list)

    # ---- SAVE CSV ----
    df_exp = pd.DataFrame(explanation_list, columns=["feature", "importance"])
    df_exp.to_csv(
        f"./src/results/lime_explanations/csv/lime_{class_name}.csv",
        index=False
    )
    print(f"LIME CSV saved: lime_{class_name}.csv")

    # ---- SAVE FIGURE ----
    plt.figure(figsize=(12, 7))
    fig = exp.as_pyplot_figure()

    plt.title(f"LIME explanation – Class {class_name}", fontsize=14)
    plt.xlabel("Feature importance", fontsize=12)
    plt.tight_layout()

    fig.savefig(f"./src/results/lime_explanations/lime_{class_name}.png")
    plt.close()

    print(f"LIME plot saved: lime_{class_name}.png")
