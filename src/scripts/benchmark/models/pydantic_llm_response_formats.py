from pydantic import BaseModel, Field, ValidationError
from typing import Literal


class NASConfig(BaseModel):
    num_blocks: Literal[2, 5, 10]
    initial_filters: Literal[16, 32, 64]
    kernel_size: Literal[3, 5]
    layers_per_block: Literal[1, 2]
    use_pooling: bool

    common_dim: Literal[64, 128, 256, 512]

    attention_mechanism: Literal[
        "no-metadata",
        "concatenation",
        "crossattention",
        "metablock"
    ]

    num_layers_text_fc: Literal[1, 2, 3]
    neurons_per_layer_size_of_text_fc: Literal[64, 128, 256, 512]

    num_layers_fc_module: Literal[1, 2]
    neurons_per_layer_size_of_fc_module: Literal[256, 512]
