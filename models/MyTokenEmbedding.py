import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

class MyTokenEmbedding(nn.Module):
    def __init__(self, patch_size, d_model, batch_size, cls=False):
        super(MyTokenEmbedding, self).__init__()
        # max_len is the maximum sequence length the model is expected to handle.
        # Each position up to max_len will have its own learnable embedding vector.
        # self.cls_token = nn.Parameter(torch.randn(batch_size, 1, 3 * patch_size * patch_size))

        self.projection = nn.Linear(3 * patch_size * patch_size, d_model)
        self.add_cls = cls
        self.patch_size = patch_size
        # We don't need to manually initialize these; nn.Embedding does it by default.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = nn.Parameter(torch.randn(x.shape[0], 1, 3 * self.patch_size * self.patch_size)).cuda()
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        if self.add_cls == True:
            y = torch.cat([y, cls_token], dim = 1)
        out = self.projection(y)
        return out