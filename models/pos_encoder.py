import torch
import torch.nn as nn
import torch.nn.functional as F


# Relative Positional Encoding, https://arxiv.org/pdf/1803.02155
# Reference: https://github.com/evelinehong/Transformer_Relative_Position_PyTorch

class RelativePositionalEncoding(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_length = max_seq_len
        self.pe = nn.Parameter(torch.randn(2*self.max_length+1, head_dim) / head_dim ** 0.5)
        
    def forward(self, seq_len):
        range_vec = torch.arange(seq_len)
        distances = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        distances = distances.clamp(-self.max_length, self.max_length) + self.max_length
        return self.pe[distances]

# See https://pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
# TODO: The following implementation is incorrect.
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10_000):
        super().__init__()
        self.dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len):
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i,j -> ij", seq_idx, self.theta).float() # (T, D/2)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1) # (T,D/2,2)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x): # (B,T,H,D)
        seq_len, head_dim = x.size(1), x.size(3)
        rope_cache = self.cache[:seq_len] # (T,D/2,2)
        x_reshape = x.float().reshape(*x.shape[:-1],-1,2) # (B,T,H,D/2,2)
        x1, x2 = x_reshape.unbind(dim=-1) # (B,T,H,D/2,1), (B,T,H,D/2,1)
        rope_cache = rope_cache.view(1,seq_len,1,head_dim//2,2) # (1,T,1,D/2,2)
        cosin, sine = rope_cache.unbind(dim=-1) # (1,T,1,D/2,1), (1,T,1,D/2,1)
        x_out = torch.stack([x1*cosin - x2*sine, x2*cosin + x1*sine], -1) # (B,T,H,D/2,2)
        x_out = x_out.flatten(3) # (B,T,H,D)
        return x_out.type_as(x)

# https://arxiv.org/pdf/2108.12409
# See https://github.com/jaketae/alibi/blob/main/alibi/attention.py
class AliBiPositionalEncoding(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        base = (2**8) ** (1 / num_heads)
        slopes = 1. / base ** torch.arange(1,num_heads+1)
        slopes = slopes.unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("slopes", slopes, persistent=False)
        
    def forward(self, seq_len):
        device = self.slopes.device
        range_vec = torch.arange(seq_len).to(device)
        distances = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        alibi_emb = self.slopes * distances.unsqueeze(0)
        return alibi_emb.unsqueeze(0)