from ml_collections import ConfigDict
import torch
import os


def get_config() -> ConfigDict:
    config = ConfigDict()

    
    config.vocab_size = 16
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.work_dir = "results"  # Specify working directory
    config.batch_size = 256
    config.test_size = 4096

    config.task = ConfigDict()
    config.task.max_variables = 12
    
    
    config.model = ConfigDict()
    config.model.emb_dim = 64
    config.model.bias = False
    config.model.mlp_bias = True
    config.model.ff_dim = 2*64
    config.model.num_layers = 2
    config.model.num_heads = (1, 1)  # Tuple of number of heads for each layer
    config.model.dropout = None  # Dropout rate, None means no dropout
    config.model.mask = True  # Whether to use masking in attention
    config.model.mlp = (False, True)  # Tuple indicating whether to use MLP in each layer
    config.model.layer_norm = False  # Whether to use layer normalization
    config.model.activation = (False, True)  # Tuple indicating whether to use activation in each layer
    config.model.pos_enc = "rotary"  # Type of positional encoding
    config.model.pos_max_len = 256  # Maximum length for positional encoding
    config.model.flash = True  # Whether to use flash attention for faster computation

    config.training = ConfigDict()
    config.training.optimizer = "adam"
    config.training.lr = 1e-3
    config.training.schedule = "triangle"
    config.training.warmup_steps = 5_000
    config.training.total_steps = 10_000
    config.training.eval_iter = 100
    config.training.get_attn = 1000
    config.training.get_checkpoints = 500
    config.training.weight_decay = 1e-2
    config.training.label_smoothing = 0.1
    config.training.freeze_value = False
    config.training.freeze_out = False
    config.training.identity_query = False

    config.wandb = ConfigDict()
    config.wandb.project = "Reasoning"  # Specify wandb project

    return config