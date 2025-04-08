from dataclasses import dataclass
import logging
import random
import math
import numpy as np
from collections import namedtuple
import pickle
import time
import torch
import sys

from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Optional, Tuple
from models.pos_encoder import *
from models.attention import *


class TFBlock(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.MHA = MultiHeadAttention(config, layer)
        self.ln1 = nn.LayerNorm(config.model.emb_dim) if config.model.layer_norm else nn.Identity()
        self.mlp = None
        self.attn_dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None

        if config.model.mlp[layer]:
            if config.model.activation[layer]:
                assert config.model.ff_dim is not None, "FeedForward dimension cannot be empty."
                self.mlp = nn.Sequential(
                    nn.Linear(config.model.emb_dim, config.model.ff_dim, bias=config.model.mlp_bias),
                    nn.ReLU(),
                    nn.Linear(config.model.ff_dim, config.model.emb_dim, bias=config.model.mlp_bias)
                )
            else:
                self.mlp = nn.Linear(config.model.emb_dim, config.model.emb_dim)
            self.ln2 = nn.LayerNorm(config.model.emb_dim) if config.model.layer_norm else nn.Identity()
            self.mlp_dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None

    def forward(self, x, get_attn=False):
        x = self.ln1(x)
        attn_map = -1
        atten_out, attn_map = self.MHA(x, get_attn)
        x = x + self.attn_dropout(atten_out) if self.attn_dropout is not None else x + atten_out
        if self.mlp is not None:
            mlp_out = self.mlp(self.ln2(x))
            x = x + self.mlp_dropout(mlp_out) if self.mlp_dropout is not None else x + mlp_out
        return x, attn_map 
        
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.model.emb_dim).to(config.device)
        self.pos_enc = config.model.pos_enc
        seq_len = config.task.max_variables * 8 - 14
        if config.model.pos_enc == "abs":
            self.positional_encoding = nn.Embedding(seq_len, config.model.emb_dim)
        self.layers = nn.ModuleList([TFBlock(config, layer) for layer in range(config.model.num_layers)])
        self.output_layer = nn.Linear(config.model.emb_dim, config.vocab_size)
        self.atten_maps = {l: torch.zeros((config.model.num_heads[l], seq_len, seq_len), device=config.device) for l in range(config.model.num_layers)}

    def forward(self, x, get_attn=False):
        if self.pos_enc == "abs":
            x = self.embed(x) + self.positional_encoding(torch.arange(x.size(1), device=x.device).view(1, x.size(1)))
        else:
            x = self.embed(x)
        for i, layer in enumerate(self.layers):
            x, attn_map = layer(x, get_attn)
            if torch.is_tensor(attn_map):
                self.atten_maps[i] = attn_map.mean(dim=0)
            
        logits = self.output_layer(x) # (batch_size, seq_len, vocab_size)
        return logits, self.atten_maps