import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pairwise_degradation_curves(
    results_directory: str = "./src/results",
    compared_methods: list[str] | None = None
):
    """
    Plota curvas de degradação comparando múltiplos mecanismos
    no mesmo gráfico para cada backbone.

    Parameters
    ----------
    results_directory : str
        Diretório contendo os arquivos summary_*.csv
    compared_methods : list[str]
        Lista com os mecanismos a serem comparados.
    """

    if compared_methods is None:
        raise ValueError(
            "compared_methods must be a list of mechanisms to compare."
        )

    search_pattern = os.path.join(results_directory, "summary_*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"❌ No files found for pattern: {search_pattern}")
        return

    all_data = []
    for file in csv_files:
        df = pd.read_csv(file)
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    required_columns = {"missing_rate", "backbone", "mechanism", "bacc_mean", "bacc_std"}
    missing_columns = required_columns - set(full_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in CSV files: {sorted(missing_columns)}"
        )

    # Garantir tipos corretos
    full_df["missing_rate"] = pd.to_numeric(full_df["missing_rate"], errors="coerce")
    full_df["bacc_mean"] = pd.to_numeric(full_df["bacc_mean"], errors="coerce")
    full_df["bacc_std"] = pd.to_numeric(full_df["bacc_std"], errors="coerce")

    full_df = full_df.dropna(subset=["missing_rate", "bacc_mean", "bacc_std"])
    full_df = full_df.sort_values(["backbone", "missing_rate"])

    missing_rates_sorted = sorted(full_df["missing_rate"].unique())
    backbones = sorted(full_df["backbone"].unique())

    colors = sns.color_palette("tab10", len(compared_methods))
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    for backbone in backbones:
        sns.set_theme(style="whitegrid", rc={"grid.linestyle": ":"})
        plt.figure(figsize=(10, 6), dpi=400)

        backbone_df = full_df[full_df["backbone"] == backbone]

        for i, mech in enumerate(compared_methods):
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]

            subset = backbone_df[backbone_df["mechanism"] == mech].copy()

            if subset.empty:
                print(f"⚠️ Mechanism '{mech}' not found for backbone '{backbone}'.")
                continue

            subset = subset.sort_values("missing_rate")

            x = subset["missing_rate"].values
            y = subset["bacc_mean"].values
            std = subset["bacc_std"].values

            plt.plot(
                x,
                y,
                marker="o",
                linestyle=linestyle,
                color=color,
                linewidth=2.5,
                label=mech,
                markersize=6
            )

            plt.fill_between(
                x,
                y - std,
                y + std,
                color=color,
                alpha=0.12
            )

        plt.title(
            f"Degradation Curves Comparison Across Mechanisms\nBackbone: {backbone.upper()}",
            fontsize=15,
            fontweight="bold",
            pad=20
        )
        plt.xlabel("Missing Metadata Rate ($\\sigma$)", fontsize=12)
        plt.ylabel("Balanced Accuracy (Mean ± STD) (5-Fold)", fontsize=12)

        plt.xticks(missing_rates_sorted)
        plt.ylim(0.25, 0.90)

        plt.legend(
            title="Mechanisms",
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
            f"BACC_degradation_{backbone}_methods_comparison.png"
        )
        plt.savefig(output_img, bbox_inches="tight", dpi=400)
        plt.close()

        print(f"📈 Saved: {output_img}")


if __name__ == "__main__":
    target_path = (
        "./src/results/testes-da-implementacao-final_2/02042026-WITH-LN--METHOD-CONFIG-COMPARISON/unfrozen_weights/8/summary"
    )

    compared_methods = [
        "no-metadata",
        "concatenation",
        "metablock",
        "crossattention",
        "att-intramodal+residual+cross-attention-metadados"
    ]

    plot_pairwise_degradation_curves(
        results_directory=target_path,
        compared_methods=compared_methods
    )