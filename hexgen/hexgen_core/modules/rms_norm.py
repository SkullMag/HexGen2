"""Added 2026-10-19: unfused RMSNorm for the explicitly adapted FABRIC pilot."""
import torch
from torch import nn


class TorchRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight
