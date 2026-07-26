import os
import numpy as np
import pandas as pd
import stats  # módulo local com statistical_test


# =====================================================
# Utils
# =====================================================
def load_dataset(file_path):
    """Loads dataset from the given file path"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def get_metric_values(
    file_folder_path,
    file_name="model_metrics.csv",
    metric_names=("accuracy", "balanced_accuracy", "f1_score", "auc")
):
    """
    Retrieves metric values from CSV.
    Returns a 1D numpy array with all requested metrics concatenated.
    """
    csv_file = os.path.join(file_folder_path, file_name)
    collected = []

    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return None

    data = load_dataset(csv_file)
    if data is None:
        return None

    for metric in metric_names:
        if metric in data.columns:
            collected.append(data[metric].values)
        else:
            print(f"Metric '{metric}' not found in {csv_file}")

    if len(collected) == 0:
        return None

    return np.concatenate(collected)


def save_statistics_tests(test_results, file_path_prefix):
    """
    Save statistical test results to CSV.
    """
    try:
        df = pd.DataFrame(test_results)
        out_path = f"{file_path_prefix}_statistics_tests_results.csv"
        df.to_csv(out_path, index=False)
        print(f"Statistics saved to {out_path}")
    except Exception as e:
        print(f"Error saving statistics: {e}")


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":

    base_file_folder_path = (
        "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/"
        "src/tests/results"
    )

    wanted_metric_list = [
        "accuracy",
        "balanced_accuracy",
        "f1_score",
        "auc"
    ]

    # Nosso método (baseline principal)
    base_model_path = (
        # "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/"
        # "src/results/testes-da-implementacao-final_2/01012026/"
        # "PAD-UFES-20/unfrozen_weights/8/"
        # "att-intramodal+residual+cross-attention-metadados/"
        # "model_caformer_b36.sail_in22k_ft_in1k_with_one-hot-encoder_512_with_best_architecture"
        "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/NAS/benchmark_nas_llm-as-controller_trainning-optimized-model-architectures/PAD-UFES-20/unfrozen_weights/8/model_nas_multimodal_model_id-21_with_one-hot-encoder_512_with_best_architecture"
    )

    list_of_used_algs = []
    list_all_models_metrics = []

    our_metrics = get_metric_values(
        file_folder_path=base_model_path,
        metric_names=wanted_metric_list
    )
    if our_metrics is not None:
        list_of_used_algs.append("our-method")
        list_all_models_metrics.append(our_metrics)

    # =====================================================
    # Baselines
    # =====================================================
    lista_modelos_e_backbones = {
        "no-metadata": "resnet-50",
        "concatenation": "resnet-50",
        # "metanet": "resnet-50",
        # "md-net": "resnet-50",
        "metablock": "efficientnet-b0",
        "crossattention": "densenet169"
#        "liwterm": "vit_large_patch16_224"
    }

    for alg, backbone in lista_modelos_e_backbones.items():
        # model_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/testes-da-implementacao-final_2/01012026/PAD-UFES-20/unfrozen_weights/8/{alg}/model_{backbone}_with_one-hot-encoder_512_with_best_architecture"
        model_path = f"/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/src/results/PAD-UFES-20/NAS/benchmark_nas_llm-as-controller_trainning-optimized-model-architectures/PAD-UFES-20/unfrozen_weights/8/{alg}/model_{backbone}_with_one-hot-encoder_512_with_best_architecture"
        metrics = get_metric_values(
            file_folder_path=model_path,
            metric_names=wanted_metric_list
        )

        if metrics is not None:
            list_of_used_algs.append(alg)
            list_all_models_metrics.append(metrics)
        else:
            print(f"Metrics not found for {alg}")

    # =====================================================
    # Statistical test
    # =====================================================
    out = stats.statistical_test(
        data=list_all_models_metrics,
        alg_names=list_of_used_algs
    )
    print(out)
    # =====================================================
    # Save results
    # =====================================================
    save_statistics_tests(
        test_results=out,
        file_path_prefix=base_file_folder_path
    )
