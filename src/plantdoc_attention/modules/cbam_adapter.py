"""CBAM channel/spatial control with the same normalization and residual gate as BRA.

Independent implementation of the sequential channel/spatial mechanism from
Woo et al., ECCV 2018. Local variant omits spatial BatchNorm; see Phase 4 guide.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .residual_attention import ChannelLayerNorm


class CBAM(nn.Module):
    def __init__(self, dim: int, reduction_ratio: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        for name, value in (("dim", dim), ("reduction_ratio", reduction_ratio), ("spatial_kernel", spatial_kernel)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if spatial_kernel % 2 == 0:
            raise ValueError("spatial_kernel must be odd")
        self.dim = dim
        hidden = max(1, dim // reduction_ratio)
        self.channel_mlp = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, dim, 1))
        self.spatial = nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.dim or min(x.shape) <= 0:
            raise ValueError(f"Expected nonempty [B,{self.dim},H,W], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("CBAM requires floating-point input")
        avg = F.adaptive_avg_pool2d(x, 1)
        maximum = F.adaptive_max_pool2d(x, 1)
        channel_gate = (self.channel_mlp(avg) + self.channel_mlp(maximum)).sigmoid()
        channel_output = x * channel_gate.to(x.dtype)
        pooled = torch.cat((channel_output.mean(1, keepdim=True),
                            channel_output.amax(1, keepdim=True)), dim=1)
        spatial_gate = self.spatial(pooled).sigmoid()
        return channel_output * spatial_gate.to(x.dtype)


class ResidualCBAM(nn.Module):
    def __init__(self, dim: int, reduction_ratio: int = 16, spatial_kernel: int = 7,
                 gamma_init: float = 0.0) -> None:
        super().__init__()
        if not math.isfinite(gamma_init):
            raise ValueError("gamma_init must be finite")
        self.attention = CBAM(dim, reduction_ratio, spatial_kernel)
        self.norm = ChannelLayerNorm(dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.gamma.to(x.dtype) * self.attention(self.norm(x))
