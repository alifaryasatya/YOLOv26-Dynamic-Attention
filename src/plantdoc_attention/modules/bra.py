"""Bi-level routing attention with NCHW input and masked spatial padding.

Independent implementation based on BiFormer (Zhu et al., CVPR 2023).
References and intentional differences are documented in docs/PHASE_1_BRA.md.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _partition(x: Tensor, region_height: int, region_width: int) -> Tensor:
    """[B,C,Hp,Wp] -> [B,R,T,C], regions/tokens in row-major order."""
    batch, channels, height, width = x.shape
    grid_height, grid_width = height // region_height, width // region_width
    return (
        x.reshape(batch, channels, grid_height, region_height, grid_width, region_width)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch, grid_height * grid_width, region_height * region_width, channels)
    )


def _unpartition(x: Tensor, grid_height: int, grid_width: int,
                 region_height: int, region_width: int) -> Tensor:
    """[B,R,T,C] -> [B,C,Hp,Wp], inverse of _partition."""
    batch, _, _, channels = x.shape
    return (
        x.reshape(batch, grid_height, grid_width, region_height, region_width, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch, channels, grid_height * region_height, grid_width * region_width)
    )


class BiLevelRoutingAttention(nn.Module):
    """Content-dependent region routing followed by attention to selected tokens.

    Args:
        dim: Input/output channel count.
        num_heads: Number of token attention heads, dividing dim and qk_dim.
        n_win: Maximum number of regions per spatial axis.
        topk: Number of selected key regions, capped to actual region count.
        qk_dim: Total Q/K channels (default dim); V retains dim channels.
        side_dwconv: Odd depthwise local-context kernel, or 0 to disable.
        attn_drop: Dropout probability applied to token attention probabilities.

    Input/output: floating Tensor[B,dim,H,W], preserving shape/dtype/device.
    Optional return_routing returns (output, indices[B,R,K]) for inspection.
    Routing is shared across heads and detached, matching official hard routing.
    """

    def __init__(self, dim: int, num_heads: int = 4, n_win: int = 4,
                 topk: int = 4, qk_dim: int | None = None,
                 side_dwconv: int = 3, attn_drop: float = 0.0) -> None:
        super().__init__()
        qk_dim = dim if qk_dim is None else qk_dim
        for name, value in (("dim", dim), ("num_heads", num_heads), ("n_win", n_win),
                            ("topk", topk), ("qk_dim", qk_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dim % num_heads or qk_dim % num_heads:
            raise ValueError("dim and qk_dim must be divisible by num_heads")
        if topk > n_win * n_win:
            raise ValueError("topk must not exceed n_win squared")
        if not isinstance(side_dwconv, int) or side_dwconv < 0 or (side_dwconv and side_dwconv % 2 == 0):
            raise ValueError("side_dwconv must be 0 or a positive odd integer")
        if not 0.0 <= attn_drop < 1.0:
            raise ValueError("attn_drop must be in [0, 1)")
        self.dim, self.qk_dim = dim, qk_dim
        self.num_heads, self.n_win, self.topk = num_heads, n_win, topk
        self.scale = (qk_dim // num_heads) ** -0.5
        self.qkv = nn.Conv2d(dim, 2 * qk_dim + dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.local_context = (
            nn.Conv2d(dim, dim, side_dwconv, padding=side_dwconv // 2, groups=dim)
            if side_dwconv else None
        )
        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, x: Tensor, *, return_routing: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1] != self.dim:
            raise ValueError(f"Expected [B,{self.dim},H,W], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("BRA requires floating-point input")
        batch, _, height, width = x.shape
        if batch <= 0 or height <= 0 or width <= 0:
            raise ValueError("Batch and spatial dimensions must be positive")
        rh, rw = math.ceil(height / self.n_win), math.ceil(width / self.n_win)
        gh, gw = math.ceil(height / rh), math.ceil(width / rw)
        pad = (0, gw * rw - width, 0, gh * rh - height)
        q, k, v = self.qkv(x).split((self.qk_dim, self.qk_dim, self.dim), dim=1)
        qt, kt, vt = (_partition(F.pad(t, pad), rh, rw) for t in (q, k, v))
        valid = _partition(F.pad(x.new_ones((1, 1, height, width)), pad), rh, rw)[..., 0].bool()
        regions, tokens = qt.shape[1:3]

        # Keep logits and routing in FP32 under AMP; FP64 remains FP64 for gradcheck.
        compute_dtype = torch.float64 if qt.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            qf, kf, vf = (t.to(compute_dtype) for t in (qt, kt, vt))
            weights = valid.to(compute_dtype).unsqueeze(-1)
            denominator = weights.sum(dim=2).clamp_min(1)
            qr = (qf.detach() * weights).sum(dim=2) / denominator
            kr = (kf.detach() * weights).sum(dim=2) / denominator
            routing_scores = (qr @ kr.transpose(-1, -2)) * self.qk_dim ** -0.5
            indices = routing_scores.topk(min(self.topk, regions), dim=-1).indices
            batch_indices = torch.arange(batch, device=x.device)[:, None, None]
            # Gather only selected regions, never an R x R x T x D tensor.
            kg = kf[batch_indices, indices].flatten(2, 3)
            vg = vf[batch_indices, indices].flatten(2, 3)
            key_valid = valid.expand(batch, -1, -1)[batch_indices, indices].flatten(2, 3)
            selected_tokens = kg.shape[2]
            qh = qf.reshape(batch, regions, tokens, self.num_heads, -1).transpose(2, 3)
            kh = kg.reshape(batch, regions, selected_tokens, self.num_heads, -1).transpose(2, 3)
            vh = vg.reshape(batch, regions, selected_tokens, self.num_heads, -1).transpose(2, 3)
            logits = (qh * self.scale) @ kh.transpose(-1, -2)
            logits = logits.masked_fill(~key_valid[:, :, None, None, :], float("-inf"))
            probabilities = self.attn_dropout(logits.softmax(dim=-1))
            attended = (probabilities @ vh).transpose(2, 3).reshape(batch, regions, tokens, self.dim)
        output = _unpartition(attended.to(v.dtype), gh, gw, rh, rw)[..., :height, :width]
        if self.local_context is not None:
            output = output + self.local_context(v)
        output = self.proj(output).to(x.dtype)
        return (output, indices) if return_routing else output
