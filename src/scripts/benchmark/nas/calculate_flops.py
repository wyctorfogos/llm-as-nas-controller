import os
import sys
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from models import dynamicMultimodalmodel


# =========================
# Config
# =========================

config = {
    "num_blocks": 10,
    "initial_filters": 64,
    "kernel_size": 3,
    "layers_per_block": 2,
    "use_pooling": True,
    "common_dim": 512,
    "attention_mechanism": "metablock",
    "num_layers_text_fc": 3,
    "neurons_per_layer_size_of_text_fc": 512,
    "num_layers_fc_module": 2,
    "neurons_per_layer_size_of_fc_module": 512,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

common_dim = config["common_dim"]
attention_mechanism = config["attention_mechanism"]

# Ajuste para o número real de features do seu OHE/metadata
num_metadata_features = 91

model_weights_path = "./src/results/PAD-UFES-20/NAS/benchmark_nas_llm-as-controller_trainning-optimized-model-architectures/PAD-UFES-20/multiclass/unfrozen_weights/8/metablock/model_nas_multimodal_model_id-0_with_one-hot-encoder_512_with_best_architecture/nas_multimodal_model_id-0_fold_3/model.pth"


# =========================
# Model
# =========================

model = dynamicMultimodalmodel.DynamicCNN(
    config=config,
    num_classes=6,
    device=device,
    common_dim=common_dim,
    num_heads=8,
    vocab_size=num_metadata_features,
    attention_mecanism=attention_mechanism,
    n=1 if attention_mechanism == "no-metadata" else 2,
)

# state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)
#model.load_state_dict(state_dict)

model.to(device)
model.eval()


# =========================
# Dummy inputs
# =========================

image = torch.randn(1, 3, 224, 224).to(device)
metadata = torch.randn(1, 91).to(device)


# =========================
# FLOPs
# =========================

with torch.no_grad():
    flops = FlopCountAnalysis(model, (image, metadata))

    total_flops = flops.total()
    total_gflops = total_flops / 1e9

    print("=" * 60)
    print("FLOPs analysis")
    print("=" * 60)
    print(f"FLOPs:  {total_flops:,}")
    print(f"GFLOPs: {total_gflops:.4f}")
    print()

    print("=" * 60)
    print("Parameters")
    print("=" * 60)
    print(parameter_count_table(model))

    print("=" * 60)
    print("Unsupported ops")
    print("=" * 60)
    print(flops.unsupported_ops())