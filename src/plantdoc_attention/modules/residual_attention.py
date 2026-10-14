"""Shape-preserving residual attention with an initially closed gate."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bra import BiLevelRoutingAttention


class ChannelLayerNorm(nn.Module):
    """LayerNorm across channels at each pixel, independent of image size."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized = F.layer_norm(
                x.permute(0, 2, 3, 1).to(dtype),
                (self.weight.numel(),), self.weight.to(dtype), self.bias.to(dtype), self.eps,
            )
        return normalized.permute(0, 3, 1, 2).contiguous().to(x.dtype)


class ResidualAttention(nn.Module):
    """y = x + gamma * BRA(ChannelLayerNorm(x)); gamma defaults to zero.

    This is a tensor adapter. YOLO graph metadata/wrapping belongs to Phase 2.
    With gamma=0 branch parameter gradients are zero on the first backward;
    gamma itself can learn, unlocking the branch in subsequent optimizer steps.
    """

    def __init__(self, dim: int, *, gamma_init: float = 0.0, **bra_kwargs: object) -> None:
        super().__init__()
        if not torch.isfinite(torch.tensor(gamma_init)):
            raise ValueError("gamma_init must be finite")
        self.attention = BiLevelRoutingAttention(dim, **bra_kwargs)
        self.norm = ChannelLayerNorm(dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: Tensor) -> Tensor:
        update = self.attention(self.norm(x))
        return x + self.gamma.to(x.dtype) * update
