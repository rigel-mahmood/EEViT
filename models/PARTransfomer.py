from math import gcd, ceil
import functools

import torch
from torch import nn, einsum
import torch.nn.functional as F

from Rotary_Embedding_torch import RotaryEmbedding, apply_rotary_emb

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import numpy as np


# -----------helper functions---------------
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class PARAttention(nn.Module):
    def __init__(
        self,
        *,
        latent_len=32,
        dim = 512, # embedding size
        heads = 8,
        causal = True,
        sequence_len = 1024,
        segment_len = 256,  # context = sequence length - latent
        num_segments,      # number of segments, the context should be divided into
        pos_emb = None,
        dropout = 0.,
        layer_num = 0,
        apply_pe_all_layers=False
    ):
        super().__init__()
        self.latent_len = latent_len
        self.sequence_len = sequence_len
        self.segment_len = segment_len
        self.num_segments = num_segments
        self. apply_pe_all_layers =  apply_pe_all_layers
        self.segment_size = segment_len # should divide evenly
        # For example, if context_len = 1536, and num_segments = 2, then
        # segment_size = 768
        assert (self.segment_len * num_segments == sequence_len), 'segment_length times num_segments should be equal to sequence length'
        self.layer_num = layer_num # for PerceiverAR, first layer is different
        self.dim_head = dim//heads  # e.g., 512/8 = 64
        self.scale = self.dim_head ** -0.5  # 1/8

        self.heads = heads
        self.causal = causal

        self.norm = nn.LayerNorm(self.dim_head)  # (64)

        self.pos_emb = default(pos_emb, RotaryEmbedding(self.dim_head))
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias = False)  # (512, 512)
        self.to_kv = nn.Linear(dim, dim, bias = False)    # (512, 512)
        self.to_out = nn.Linear(dim, dim) # (512, 512)
        self.to_out_0 = nn.Linear(dim, dim) # (512, 512)

       


    def forward(self, x, mask = None): # e.g., x has shape of (b, n, d)
        b, n, *_, h, device, causal = *x.shape, self.heads, x.device, self.causal, 

        mask_value = -torch.finfo(x.dtype).max
        context_len = x.shape[1] - self.latent_len
        if self.layer_num == 0:
            x_latent = x[:,context_len:,:] 
        else:
            x_latent = x

        qkv = (self.to_q(x), self.to_kv(x))
        padded_len = x.shape[-2]    # 1024
        padded_len_q = x_latent.shape[-2]
        # get sequence range, for calculating mask
        seq_range = torch.arange(padded_len, device = device)

        # split heads
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)  # b = 4, n = 1024, h = 8, d = 64; 
        # q, kv = [32, 1024, 64]    # (4, 1024, (8x64=512)) --> ((4x8), 1024, 64), h=8
        # rotary embedding
        if exists(self.pos_emb):
            if self.layer_num == 0:
                rotary_emb = self.pos_emb(seq_range, cache_key = padded_len) # (1024, 64)
                rotary_emb = rearrange(rotary_emb, 'n d -> () n d') # (1,1024,64)
                q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        q = q * self.scale # scale queries

        if self.layer_num == 0:
            q_latent = q[:,context_len:,:]
        else:
            q_latent = q

        lkv = self.norm(kv)
        lsim = einsum('b i j, b k j -> b i k', q_latent, kv) # [24,3,256,512] #'ij, kj -> ik

        attn= lsim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # masking
        m_size = lsim.shape[-2]  # 256

        out = einsum('b i j, b j k->b i k',attn,lkv)
        out = rearrange(out, '(b h) n d -> b (n) (h d)', h = h)

        return out


class PARTransformer(nn.Module):
    def __init__(
        self,
        *,
        latent_len,
        num_tokens,  # for output size determination
        dim,   # embedding
        segment_len,
        num_segments,
        num_layers, # number of layers
        heads = 8,
        sequence_len,
        causal = True,
        ff_mult = 4, # expand feedforward by 4 times then reduce back to embedding size
        ff_dropout = 0.,
        attn_dropout = 0.,
        apply_pe_all_layers = False
    ):
        super().__init__()
        self.latent_len = latent_len
        self.sequence_len = sequence_len
        self.segment_len = segment_len
        self.num_segments = num_segments,
       
        patch_dim = 3 * 8 * 8
        self.token_emb = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 8, p2 = 8),
            nn.Linear(patch_dim, dim),
        )  
        self.dim_head = dim//heads
        self.apply_pe_all_layers = apply_pe_all_layers
        pos_emb = RotaryEmbedding(self.dim_head)
 
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, PARAttention(dim = dim, heads = heads, latent_len=self.latent_len, sequence_len = sequence_len, segment_len = segment_len, num_segments=num_segments, 
                                            causal = causal,layer_num=i, pos_emb = pos_emb, dropout = attn_dropout, apply_pe_all_layers=apply_pe_all_layers)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x, mask = None):
        x = self.token_emb(x)
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x[:,-self.latent_len:,:]
            x = ff(x) + x   
        out = x[:,-1,:]  
        out = self.to_logits(out)
        return out
