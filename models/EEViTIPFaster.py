from math import gcd, ceil
import functools
import torch
from torch import nn, einsum
import torch.nn.functional as F
from Rotary_Embedding_torch import RotaryEmbedding, apply_rotary_emb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import fastervit

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

class EEViT_IPAttention(nn.Module):
    def __init__(
        self,
        *,
        dim = 512, # embedding size
        heads = 8,
        causal = True,
        sequence_len = 1024,
        segment_len = 256,  # context = sequence length - latent
        num_segments,      # number of segments the context should be divided into
        pos_emb = None,
        dropout = 0.,
        layer_num = 0,
        apply_pe_all_layers=False
    ):
        super().__init__()
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
       


    def forward(self, x, mask = None): 
        b, n, *_, h, device, causal = *x.shape, self.heads, x.device, self.causal, 

        mask_value = -torch.finfo(x.dtype).max

        # get queries, keys, values
        qkv = (self.to_q(x), self.to_kv(x))  # x = (4, 1024, 512)
        padded_len = x.shape[-2]    # 1024
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
            if self.layer_num > 0 and self.apply_pe_all_layers == True:
                rotary_emb = self.pos_emb(seq_range, cache_key = padded_len) # (1024, 64)
                rotary_emb = rearrange(rotary_emb, 'n d -> () n d') # (1,1024,64)
                q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        q = q * self.scale # scale queries
        qs = rearrange(q, 'c (s m) d ->c s m d', s = self.num_segments) 
        kvs = rearrange(kv,'c (s m) d ->c s m d', s = self.num_segments)
        # catenate consecutive statements for kv
        fcat = lambda t : torch.cat([kvs[:,t,:,:],kvs[:,t+1,:,:]], dim=1).unsqueeze(dim=1)
        kvs_ccs = torch.cat([fcat(t) for t in range(0,self.num_segments-1)],dim=1)
        qs_ccs = qs[:,1:self.num_segments,:,:]
        lkv = self.norm(kvs_ccs)
        lsim = einsum('c s m d, c s n d -> c s m n', qs_ccs, kvs_ccs) 
        # masking
        m_size = lsim.shape[-2]  

        attn0 = einsum('c s m d, c s n d->c s m n',qs[:,0:1,:,:],kvs[:,0:1,:,:])
         
        # final attention
        attn_0= attn0.softmax(dim=-1)
        attn_1 = lsim.softmax(dim = -1)
        attnd_0 = self.attn_dropout(attn_0)
        attnd_1 = self.attn_dropout(attn_1)

        out0 = einsum('c s i j, c s j d -> c s i d', attnd_0, kvs[:,0:1,:,:])
        out1 = einsum('c s i j, c s j d -> c s i d', attnd_1, kvs_ccs)
 
        out1 = rearrange(out1,'c s z d-> c (s z) d') # merge segments  [24,1024,128]
        out_1 = rearrange(out1, '(b h) n d -> b (n) (h d)', h = h)  # [b,1024,6x128]
        out_1o = self.to_out(out_1)
        out0 = rearrange(out0,'c s z d-> c (s z) d') # merge segments
        out_0 = rearrange(out0, '(b h) n d -> b (n) (h d)', h = h)
        out_0o = self.to_out_0(out_0)
        out = torch.cat([out_0o, out_1o], dim = 1)
        return out


# function to replace Attention modules in FasterViT
def replace_attention(model, new_attn_cls, num_segments=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Module):
            # Recursively replace in child modules
            replace_attention(module, new_attn_cls)
        if hasattr(module, "attn"):  # typical naming in ViT blocks
            old_attn = module.attn
            dim = old_attn.qkv.in_features  # get input dimension
            num_heads = getattr(old_attn, "num_heads", 8)
            module.attn = new_attn_cls(dim=dim, heads=num_heads, num_segments=num_segments)
            
