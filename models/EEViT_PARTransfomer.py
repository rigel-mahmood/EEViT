from math import gcd, ceil
import functools
import torch
from torch import nn, einsum
import torch.nn.functional as F
from Rotary_Embedding_torch import RotaryEmbedding, apply_rotary_emb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from MyTokenEmbedding import MyTokenEmbedding

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

class EEViT_PARAttention(nn.Module):
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
        apply_pe_all_layers=False,
        k_context_layers=3,
        use_cls=False,
        use_cls_token_last_n_layers=2,
        num_layers=2
    ):
        super().__init__()
        self.latent_len = latent_len
        self.sequence_len = sequence_len
        self.segment_len = segment_len
        self.num_segments = num_segments
        self. apply_pe_all_layers =  apply_pe_all_layers
        self.num_layers = num_layers
        self.segment_size = segment_len # should divide evenly
        
        assert (self.segment_len * num_segments == sequence_len), 'segment_length times num_segments should be equal to sequence length'
        self.layer_num = layer_num # for PerceiverAR, first layer is different
        self.dim_head = dim//heads  # e.g., 512/8 = 64
        self.scale = self.dim_head ** -0.5  
        self.use_cls_token_last_n_layers = use_cls_token_last_n_layers

        self.heads = heads
        self.causal = causal

        self.norm = nn.LayerNorm(self.dim_head)  

        self.pos_emb = default(pos_emb, RotaryEmbedding(self.dim_head))
        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, dim, bias = False)  
        self.to_kv = nn.Linear(dim, dim, bias = False)    
        self.to_out_context = nn.Linear(dim, dim) 
        self.to_out_latent = nn.Linear(dim, dim) 
        self.k_context_layers = k_context_layers
        self.use_cls = use_cls


    def forward(self, x, mask = None): # e.g., x has shape of (b, n, d)
        b, n, *_, h, device, causal = *x.shape, self.heads, x.device, self.causal, 
        if self.use_cls == True and self.layer_num < (self.num_layers - self.use_cls_token_last_n_layers):
            cls_token = x[:,-1:,:] 
            x = x[:,0:-1,:]
        if self.layer_num == 4:
            a = 5
        mask_value = -torch.finfo(x.dtype).max

        context_len = x.shape[1] - self.latent_len
        if self.layer_num == 0:
            x_latent = x[:,context_len:,:] 
        else:
            x_latent = x

        # get queries, keys, values
        qkv = (self.to_q(x), self.to_kv(x))
        padded_len = x.shape[-2]    
        padded_len_q = x_latent.shape[-2]
        # get sequence range, for calculating mask
        seq_range = torch.arange(padded_len, device = device)
        
        # split heads
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)  

        # rotary embedding
        if exists(self.pos_emb):
            if self.layer_num == 0:
                rotary_emb = self.pos_emb(seq_range, cache_key = padded_len) 
                rotary_emb = rearrange(rotary_emb, 'n d -> () n d') 
                q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        q = q * self.scale # scale queries

        if self.layer_num <= self.k_context_layers:
            q_latent = q[:,context_len:,:]
            q_context = q[:,0:context_len,:]
            kv_context = kv[:,0:context_len,:]
        else:
            q_latent = q
            kv_context = kv

        lkv = self.norm(kv)

        if self.layer_num < self.k_context_layers:
            lkv_context = self.norm(kv_context)

        lsim_latent = einsum('b i j, b k j -> b i k', q_latent, kv) 

        if self.layer_num < self.k_context_layers:
            lsim_context = einsum('b i j, b k j -> b i k', q_context, kv_context) 

        attn_latent= lsim_latent.softmax(dim=-1)
        attn_latent = self.attn_dropout(attn_latent)

        if self.layer_num < self.k_context_layers:
            attn_context= lsim_context.softmax(dim=-1)
            attn_context = self.attn_dropout(attn_context)

        out_latent = einsum('b i j, b j k->b i k',attn_latent,lkv)
        out_latent = rearrange(out_latent, '(b h) n d -> b (n) (h d)', h = h)

        if self.layer_num < self.k_context_layers:
            out_context = einsum('b i j, b j k->b i k',attn_context,lkv_context)
            out_context = rearrange(out_context, '(b h) n d -> b (n) (h d)', h = h)
            out_context = self.to_out_context(out_context)

        out_latent = self.to_out_latent(out_latent)

        if self.layer_num < self.k_context_layers:
            out = torch.cat([out_context, out_latent], dim = 1)
        else:
            out = out_latent

        if self.use_cls == True and self.layer_num < (self.num_layers - self.use_cls_token_last_n_layers):
            out = torch.cat([out, cls_token], dim = 1) 
        return out


class EEViT_PARTransformer(nn.Module):
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
        apply_pe_all_layers = False,
        k_context_layers = 3,
        use_cls = False,
        batch_size = 64,
        patch_size = 4,
        use_cls_token_last_n_layers=2
    ):
        super().__init__()
        self.latent_len = latent_len
        self.sequence_len = sequence_len
        self.segment_len = segment_len
        self.patch_size = patch_size
        self.use_cls_token_last_n_layers = use_cls_token_last_n_layers
        self.num_segments = num_segments,
        self.num_layers=num_layers

        patch_dim = 3 * 4 * 4
        self.use_cls = use_cls
        self.batch_size = batch_size
        if self.use_cls == True:
            self.token_emb = MyTokenEmbedding(patch_size=self.patch_size, d_model=dim, batch_size=self.batch_size, cls=self.use_cls)
        else:
            self.token_emb = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 4, p2 = 4),
                nn.Linear(patch_dim, dim),
            )
           
        self.dim_head = dim//heads
        self.apply_pe_all_layers = apply_pe_all_layers
        pos_emb = RotaryEmbedding(self.dim_head)
        self.k_context_layers = k_context_layers
 
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, EEViT_PARAttention(dim = dim, heads = heads, latent_len=self.latent_len, sequence_len = sequence_len, segment_len = segment_len, num_segments=num_segments, 
                                            causal = causal,layer_num=i, pos_emb = pos_emb, dropout = attn_dropout, apply_pe_all_layers=apply_pe_all_layers,k_context_layers=self.k_context_layers,use_cls=self.use_cls, use_cls_token_last_n_layers=self.use_cls_token_last_n_layers, num_layers=self.num_layers)),
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
            if (attn.fn.layer_num >= self.k_context_layers) and (attn.fn.layer_num < (self.num_layers - self.use_cls_token_last_n_layers)):
                zz = attn(x, mask = mask)
                yy = x[:,-self.latent_len-1:-1,:]
                if self.use_cls:
                    cls_token = x[:,-1:,:]
                    x = attn(x, mask = mask)[:,-self.latent_len-1:-1,:] + x[:,-self.latent_len-1:-1,:]
                    x = torch.cat([x, cls_token], dim=1)
                else:
                    x = attn(x, mask = mask) + x[:,-self.latent_len:,:]
                
            else:
                x = attn(x, mask = mask) + x
            x = ff(x) + x   
        out = x[:,-1,:]  
        out = self.to_logits(out)
        return out 
