import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

dataset_name = "PAD-UFES-20"
# Path to your CSV file
## csv_file_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes/testes-da-implementacao-final/differents_dimension_of_projected_features/{dataset_name}/unfrozen_weights/8/all_metric_values.csv"
#csv_file_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/{dataset_name}/unfrozen_weights/8/all_metric_values.csv"
csv_file_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/01012026/different_features_with_dimension_size/PAD-UFES-20/unfrozen_weights/8/all_metric_values.csv"
# Output directory for plots
output_dir = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/01012026/different_features_with_dimension_size/PAD-UFES-20/bacc_plots"
os.makedirs(output_dir, exist_ok=True)

# Load the CSV
df = pd.read_csv(csv_file_path)

# Extract mean and std from 'balanced_accuracy' (assumes format like '0.4202 ± 0.0341')
def extract_stats(value):
    match = re.match(r'([0-9.]+)\s*±\s*([0-9.]+)', value)
    if match:
        return float(match.group(1)), float(match.group(2))
    return float('nan'), float('nan')

df[['BACC_mean', 'BACC_std']] = df['balanced_accuracy'].apply(lambda x: pd.Series(extract_stats(x)))

# Plot settings
sns.set(style="whitegrid")
palette = sns.color_palette("tab10")[:df['attention_mecanism'].nunique()]

# Loop through each common_size and plot
for common_size in sorted(df['common_size'].unique()):
    subset = df[df['common_size'] == common_size].sort_values("BACC_mean", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=subset,
        x='model_name',
        y='BACC_mean',
        hue='attention_mecanism',
        palette=palette,
        ci=None
    )
    
    # Add error bars
    for i, row in subset.iterrows():
        plt.errorbar(
            x=subset['model_name'].tolist().index(row['model_name']),
            y=row['BACC_mean'],
            yerr=row['BACC_std'],
            fmt='none',
            c='black',
            capsize=5
        )
    
    plt.title(f"Balanced Accuracy (BACC) for CNNs with Common Dimension {common_size}")
    plt.ylabel("Balanced Accuracy (BACC)")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.legend(title="Attention Mechanism", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"bacc_plot_common_size_{common_size}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.show()

    print(f"Saved plot: {plot_path}")
