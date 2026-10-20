"""Explicit common unfused RMSNorm used by both comparison adapters."""
import torch
from torch import nn
class RMSNorm(nn.Module):
    def __init__(self,hidden_size,eps=1e-6,device=None,dtype=None):
        super().__init__();self.weight=nn.Parameter(torch.ones(hidden_size,device=device,dtype=dtype));self.eps=eps
    def forward(self,x):
        values=x.float();return self.weight*(values*torch.rsqrt(values.square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)
