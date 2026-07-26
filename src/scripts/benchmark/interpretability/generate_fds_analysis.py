import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
INPUT_DIR = "./results/XAI"
OUT_DIR = "./results/XAI/FDS"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FROZEN = os.path.join(INPUT_DIR, "prediction_analysis_frozen_weights.csv")
CSV_UNFROZEN = os.path.join(INPUT_DIR, "prediction_analysis_unfrozen_weights.csv")

# pesos do índice
W_JS = 0.45
W_DELTA = 0.35
W_FLIP = 0.20

LOG2 = np.log(2)


# ==============================
# Função FDS
# ==============================
def compute_fds(df):
    df = df.copy()

    # normalizar JS (0 → log(2))
    df["js_norm"] = df["js_divergence"] / LOG2
    df["js_norm"] = df["js_norm"].clip(0, 1)

    # normalizar delta confiança
    df["delta_norm"] = df["delta_conf_baseline_class"] / (
        df["baseline_confidence"] + 1e-12
    )
    df["delta_norm"] = df["delta_norm"].clip(0, 1)

    # FDS
    df["FDS"] = (
        W_JS * df["js_norm"] +
        W_DELTA * df["delta_norm"] +
        W_FLIP * df["class_flip"]
    )

    return df


# ==============================
# Carregar dados
# ==============================
df_frozen = pd.read_csv(CSV_FROZEN)
df_unfrozen = pd.read_csv(CSV_UNFROZEN)

df_frozen = compute_fds(df_frozen)
df_unfrozen = compute_fds(df_unfrozen)

# remover baseline da análise
df_frozen = df_frozen[df_frozen["configuration"] != "original_metadata"]
df_unfrozen = df_unfrozen[df_unfrozen["configuration"] != "original_metadata"]

# ==============================
# Tabela consolidada
# ==============================
table = pd.DataFrame({
    "Feature": df_frozen["configuration"].values,
    "FDS_Frozen": df_frozen["FDS"].values,
    "FDS_Unfrozen": df_unfrozen["FDS"].values,
})

table["Difference (Frozen - Unfrozen)"] = (
    table["FDS_Frozen"] - table["FDS_Unfrozen"]
)

table.sort_values("FDS_Frozen", ascending=False, inplace=True)

table_path = os.path.join(OUT_DIR, "feature_dependency_table.csv")
table.to_csv(table_path, index=False)
print(f"Saved table: {table_path}")


# ==============================
# Barplot comparativo
# ==============================
plt.figure(figsize=(10, 6))

x = np.arange(len(table["Feature"]))
width = 0.35

plt.bar(x - width/2, table["FDS_Frozen"], width, label="Frozen")
plt.bar(x + width/2, table["FDS_Unfrozen"], width, label="Unfrozen")

plt.xticks(x, table["Feature"], rotation=45)
plt.ylabel("Feature Dependency Score (FDS)")
plt.title("Feature Dependency Score per Clinical Attribute")
plt.legend()
plt.tight_layout()

barplot_path = os.path.join(OUT_DIR, "fds_barplot.png")
plt.savefig(barplot_path, dpi=300)
plt.close()
print(f"Saved barplot: {barplot_path}")


# ==============================
# Heatmap (opcional)
# ==============================
heatmap_data = np.vstack([
    table["FDS_Frozen"].values,
    table["FDS_Unfrozen"].values
])

plt.figure(figsize=(10, 4))
plt.imshow(heatmap_data, aspect="auto")
plt.colorbar(label="FDS")
plt.yticks([0, 1], ["Frozen", "Unfrozen"])
plt.xticks(range(len(table["Feature"])), table["Feature"], rotation=45)
plt.title("Feature Dependency Heatmap")
plt.tight_layout()

heatmap_path = os.path.join(OUT_DIR, "fds_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"Saved heatmap: {heatmap_path}")


# ==============================
# Índice Global
# ==============================
global_frozen = table["FDS_Frozen"].mean()
global_unfrozen = table["FDS_Unfrozen"].mean()

print("\nGlobal Feature Dependency Score:")
print(f"Frozen:   {global_frozen:.4f}")
print(f"Unfrozen: {global_unfrozen:.4f}")

with open(os.path.join(OUT_DIR, "global_fds.txt"), "w") as f:
    f.write(f"Frozen: {global_frozen:.6f}\n")
    f.write(f"Unfrozen: {global_unfrozen:.6f}\n")