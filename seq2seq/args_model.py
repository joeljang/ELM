from adapters import ADAPTER_CONFIG_MAPPING
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    save_adapter_weights: bool = field(
        default=True,
        metadata={
            "help": "Save the weights for the task-specific adapter."}
    )
    load_adapter_weights: bool = field(
        default=False,
        metadata={
            "help": "Load the weights used to task-sepcific adapters."}
    )
    adapter_dir: str = field(
        default=None,
        metadata={
            "help": "Path to load task-specific adapters"}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    load_prefix_embeddings: bool = field(
        default=False,
        metadata={
            "help": "load prefix embeddings or not"
        },
    )
    save_prefix_only: bool = field(
        default=False,
        metadata={
            "help": "save prefix embeddings only"
        },
    )

    prompt_embedding_path: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of the paths to prefix embeddings"}
    )

    target_prompt_embedding_path: Optional[str] = field(
        default=None,
        metadata={"help": "a path to the target prompt embedding"}
    )
    
    attn_prefix_tuning: bool = field(
        default=False,
        metadata={
            "help": "Set true if you try ATTEMPT."
        },
    )

    attn_method: Optional[str] = field(
        default="sub",
        metadata={
            "help": "Attention model for attn_prefix. We currently support the following methods: linear, sub (our main method), and constant (gives the constant and equal weights to all of the prompts.)"
        },
    )

    shared_attn: bool = field(
        default=False,
        metadata={
            "help": "shared attention"
        },
    )

    load_attention: bool = field(
        default=False,
        metadata={
            "help": "Set true if you want to load pre-trained attention weights"
        },
    )

    attn_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to attention weights (linear attentions). "
        },
    )

    attn_path_sub: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "list of the path to attention weights (sub attentions). [path_to_down_projection_weights, path_to_up_projection_weights]"
        },
    )

    ignore_target: bool = field(
        default=False,
        metadata={
            "help": "Whether to ignore the new target tokens. Mainly for ablation."
        },
    )

    fix_attention: bool = field(
        default=False,
        metadata={
            "help": "this will make the attention weights frozen during training. Mainly for ablation."
        },
    )

    temperature: float = field(
        default=2000,
        metadata={
            "help": "set the soft max temperature of ATTEMPT."
        },
    )

    attn_learning_rate: float = field(
        default=None,
        metadata={
            "help": "set the learning rate for the attention modules."
        },
    )

    load_layer_norm: bool = field(
        default=False,
        metadata={
            "help": "Set true if you want to load pre-trained layer-norm weight and biases."
        },
    )

    layer_norm_dir: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Layer norm dir. [path_to_layer_norm_weight.pt, path_to_layer_norm_bias.pt]"
        },
    )

    prefix_num: Optional[int] = field(
        default=1, metadata={"help": "the number of prefix"})