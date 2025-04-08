import torch
import torch.nn as nn
import torch.nn.functional as F


class SAE(nn.Module):
    def __init__(self, config):
        super(SAE, self).__init__()
        hidden_dim = config.factor * config.emb_dim
        self.weight = nn.Parameter(torch.Tensor(config.emb_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.activation = nn.ReLU()
    
    def forward(self, x):
        c = F.linear(x, self.weight, self.bias)
        c = self.activation(c)
        out = F.linear(c, self.weight.t(), None)
        return out


# We want to see what input activate the neurons in the hidden layer