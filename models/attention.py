from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from models.pos_encoder import *

# TODO: Add MLA, MQA, GQA

# causal mask for flex_attention, not in use yet. 
# flex_attention is a fast implementation of multihead attention. 
# Yet it has not support positional encodings with training parameters.  

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


####################
# Rotary Embedding #
####################

# from https://github.com/JiajunSong629/ood-generalization-via-composition/blob/main/synthetic-experiments/model.py#L71

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Rotary embedding helper function"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (freqs_cis.shape, x.shape)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


######################
# MultiHeadAttention #
######################

class MultiHeadAttention(nn.Module):
    def __init__(self, config, layer=0):
        super().__init__()
        self.emb_dim = config.model.emb_dim
        self.n_head = config.model.num_heads[layer]
        self.head_dim = self.emb_dim // self.n_head
        assert self.emb_dim % self.n_head == 0, "Embedding dimension must be divisible by the number of heads."
        if config.training.identity_query:
            self.query = nn.Identity()
        else:
            self.query = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        self.key = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        self.value = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        if config.training.freeze_value:
            self.value.weight.requires_grad_(False)
        self.out = nn.Linear(self.emb_dim, self.emb_dim, bias=config.model.bias)
        if config.training.freeze_out:
            self.out.weight.requires_grad_(False)
        # self.mask = torch.tril(torch.ones((config.task.max_seq_len, config.task.max_seq_len), device=config.device)).unsqueeze(0).unsqueeze(1) # TODO: make self.mask a register_buffer
        self.get_attn = config.training.get_attn
        self.pos_enc = config.model.pos_enc
        self.scale = self.head_dim ** 0.5
        self.flash = config.model.flash
        self.dropout = config.model.dropout if config.model.dropout else 0.
        assert not (self.flash and self.pos_enc == "rpe"), "Flash Attention does not support RPE currently."  
        if self.pos_enc == "rpe":
            if not self.flash:
                self.PEV = RelativePositionalEncoding(self.head_dim, config.model.pos_max_len) # (T,T,D)
                self.PEK = RelativePositionalEncoding(self.head_dim, config.model.pos_max_len) # (T,T,D)
            elif config.device == "cuda":
                self.rpe = torch.randn((2*config.model.pos_max_len+1, self.head_dim), device=config.device) / (self.head_dim ** 0.5)
                
            else:
                raise ValueError("Flash Attention with RPE is currently only supported on CUDA devices.") # TODO: pay a closer look to flex_attention
        
        elif self.pos_enc == "rotary":
            self.freqs_cis = precompute_freqs_cis(self.head_dim, config.model.pos_max_len * 2, # config.rotary_theta,
            ).to(config.device)
        elif self.pos_enc == "alibi":
            self.alibi_emb = AliBiPositionalEncoding(self.n_head)
    

    def forward(self, x, get_attn=False, mask=None): # x: (B,T,C)
        if mask is not None:
            assert mask.dtype == torch.bool, "Mask must be a boolean tensor."
        batch_size, seq_len, _ = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        K = self.key(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)
        V = self.value(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1,2) # (B,H,T,D)

        if self.pos_enc == "rotary":
            Q, K = apply_rotary_emb(Q.transpose(1, 2), K.transpose(1, 2), freqs_cis=self.freqs_cis[:seq_len])
            Q, K = Q.transpose(1, 2), K.transpose(1, 2)
            
        if self.flash and (not get_attn):
            if mask is None:
                out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout, is_causal=True)
            else:
                out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout, attn_mask=mask)
            out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
            out = self.out(out)
            return out, -1
        else:
            attn_score = Q @ K.transpose(-1,-2) / self.scale # (B,H,T,T)
            if self.pos_enc == "rpe":
                Q2 = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(0,1) # (T,B,H,D)
                Q2 = Q2.contiguous().view(seq_len, batch_size*self.n_head, self.head_dim) # (T,BH,D)
                attn_score2 = torch.matmul(Q2, self.PEK(seq_len).transpose(1,2)) # (T,BH,D) @ (T,D,T) -> (T,BH,T)
                attn_score2 = attn_score2.view(seq_len, self.n_head, batch_size, seq_len).transpose(0,2).contiguous() # (B,H,T,T)
                attn_score += attn_score2 / self.scale
            elif self.pos_enc=="alibi":
                attn_score += self.alibi_emb(self.seq_len)
            assert mask is not None, "Mask must be provided for causal attention when not using flash attention."
            
            attn_score = attn_score.masked_fill(~mask, -float("inf"))

            attn = F.softmax(attn_score, dim=-1) # (B,H,T,T)
            out = attn @ V # (B,H,T,D)
            if self.pos_enc == "rpe":
                attn2 = attn.transpose(0,2).contiguous().view(seq_len, -1, seq_len) # (T,BH,T)
                out2 = torch.matmul(attn2, self.PEV(seq_len)) # (T,BH,T) @ (T,T,D) -> (T,BH,D)
                out2 = out2.view(seq_len, -1, batch_size, self.head_dim).transpose(0,2) # (B,H,T,D)
                out += out2
            out = out.transpose(1,2).contiguous().view(batch_size,seq_len,-1) # (B,T,C)
            out = self.out(out)
            if get_attn:
                return out, attn.detach()
            else:
                return out, -1