import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os

def plot_pairwise_degradation_curves(
        results_directory="./src/results",
        compared_methods:dict = None
    ):
    search_pattern = os.path.join(results_directory, "summary_*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"❌ No files found for pattern: {search_pattern}")
        return

    all_data = [pd.read_csv(file) for file in csv_files]
    full_df = pd.concat(all_data, ignore_index=True)

    # Sorting and ensuring data types
    full_df = full_df.sort_values("missing_rate")
    missing_rates_sorted = sorted(full_df["missing_rate"].unique())
    backbones = sorted(full_df["backbone"].unique())

    # Distinct palette for each pair
    colors = sns.color_palette("tab10", len(compared_methods))
    
    for backbone in backbones:
        sns.set_theme(style="whitegrid", rc={"grid.linestyle": ":"})
        plt.figure(figsize=(10, 6), dpi=400)
        
        backbone_df = full_df[full_df["backbone"] == backbone]

        for i, (base_mech, improved_mech) in enumerate(compared_methods.items()):
            color = colors[i]
            
            for mech, linestyle, alpha_fill in [(base_mech, "--", 0.05), (improved_mech, "-", 0.15)]:
                subset = backbone_df[backbone_df["mechanism"] == mech]

                if subset.empty:
                    continue

                x = subset["missing_rate"].values
                y = subset["bacc_mean"].values
                std = subset["bacc_std"].values

                plt.plot(x, y, marker="o", linestyle=linestyle, color=color, 
                         linewidth=2.5, label=mech, markersize=6)
                plt.fill_between(x, y - std, y + std, color=color, alpha=alpha_fill)

        # Labels and Title in English
        plt.title(
            f"Ablation Impact: Base vs. Enhanced Mechanisms\nBackbone: {backbone.upper()}",
            fontsize=15, fontweight="bold", pad=20
        )
        plt.xlabel("Missing Metadata Rate ($\sigma$)", fontsize=12)
        plt.ylabel("Balanced Accuracy (Mean ± STD) (5-Fold)", fontsize=12)
        
        plt.xticks(missing_rates_sorted)
        
        # Consistent Y-axis for easier comparison between backbones
        plt.ylim(0.25, 0.90)
        
        # Legend inside - Lower Left is typically the "empty" zone for degradation curves
        plt.legend(
            title="Architectural Pairs (Dashed: Base | Solid: Improved)", 
            loc="lower left", 
            fontsize=9, 
            title_fontsize=10,
            frameon=True,
            shadow=True,
            fancybox=True
        )
        
        plt.tight_layout()

        output_img = os.path.join(
            results_directory,
            f"BACC_pairwise_degradation_{backbone}_methods_comparison.png"
        )
        plt.savefig(output_img, bbox_inches="tight", dpi=400)
        plt.close()

        print(f"📈 Saved: {output_img}")

if __name__ == "__main__":
    target_path = "./src/results/testes-da-implementacao-final_2/02042026-WITH-LN--METHOD-CONFIG-COMPARISON/unfrozen_weights/8/summary"
    # Pairing logic: Base -> Proposed
    # compared_methods = {
    #     "att-intramodal": "att-intramodal+residual",
    #     "cross-attention-only": "residual+cross-attention-metadados",
    #     "crossattention": "att-intramodal+residual+cross-attention-metadados"
    # }
    compared_methods =  {"no-metadata", "concatenation", "metablock", "crossattention", "att-intramodal+residual+cross-attention-metadados"}
    plot_pairwise_degradation_curves(results_directory=target_path, compared_methods=compared_methods)