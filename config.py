from ml_collections import ConfigDict
import torch
import os


def get_config() -> ConfigDict:
    config = ConfigDict()

    
    config.vocab_size = 16
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.work_dir = "results"  # Specify working directory
    config.batch_size = 128
    config.test_size = 4096

    config.task = ConfigDict()
    config.task.max_variables = 12
    config.task.max_seq_len = config.task.max_variables * 8 - 13
    
    
    config.model = ConfigDict()
    config.model.emb_dim = 64
    config.model.bias = False
    config.model.mlp_bias = True
    config.model.ff_dim = 4*64
    config.model.num_layers = 2
    config.model.num_heads = tuple([1]*config.model.num_layers)  # Tuple of number of heads for each layer
    config.model.dropout = 0.1  # Dropout rate, None means no dropout
    config.model.mlp = tuple([True]*config.model.num_layers)  # Tuple indicating whether to use MLP in each layer
    config.model.layer_norm = True  # Whether to use layer normalization
    config.model.activation = tuple([True]*config.model.num_layers)  # Tuple indicating whether to use activation in each layer
    config.model.pos_enc = "rotary"  # Type of positional encoding
    config.model.pos_max_len = config.task.max_seq_len  # Maximum length for positional encoding
    config.model.flash = True  # Whether to use flash attention for faster computation

    config.training = ConfigDict()
    config.training.optimizer = "adamw"
    config.training.lr = 5e-4
    config.training.schedule = "triangle"
    config.training.warmup_steps = 20_000
    config.training.pad_ignore = False  # Whether to ignore padding in the attention calculation, set to True will significantly slow down training
    config.training.total_steps = 60_000
    config.training.eval_iter = 150
    config.training.get_attn = 0
    config.training.get_checkpoints = 500
    config.training.weight_decay = 1e-2
    config.training.label_smoothing = 0
    config.training.freeze_value = False
    config.training.freeze_out = False
    config.training.identity_query = False

    config.wandb = ConfigDict()
    config.wandb.project = "Reasoning"  # Specify wandb project

    return config