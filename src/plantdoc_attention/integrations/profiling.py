"""Full-image FLOPs estimate including BRA matrix multiplications."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import thop
from ultralytics.nn.modules.block import AAttn, Attention
from ultralytics.utils.torch_utils import _attention_ops, get_flops

from ..modules.bra import BiLevelRoutingAttention
from ..modules.cbam_adapter import CBAM
from ..modules.residual_attention import ChannelLayerNorm


def _bra_ops(module, inputs, output) -> None:
    batch, channels, height, width = inputs[0].shape
    rh, rw = math.ceil(height / module.n_win), math.ceil(width / module.n_win)
    regions = math.ceil(height / rh) * math.ceil(width / rw)
    tokens, selected = rh * rw, min(module.topk, regions)
    # THOP counts multiply-accumulate operations; final estimate doubles MACs.
    routing = batch * regions * regions * module.qk_dim
    token = batch * regions * tokens * selected * tokens * (module.qk_dim + channels)
    # Installed THOP adds a module's own rule to its child rules; convs are
    # already counted by child hooks and must not be charged again here.
    module.total_ops += routing + token


def _norm_ops(module, inputs, output) -> None:
    module.total_ops += 5 * inputs[0].numel()


def _cbam_ops(module, inputs, output) -> None:
    # Functional channel/spatial rescaling; MLP/spatial convs have child hooks.
    module.total_ops += 2 * inputs[0].numel()


def model_gflops(network, imgsz: int = 640) -> float:
    """Estimate at full image size; no stride proxy for spatial attention.

    Reports 2*MACs, including projections, BRA QK/AV and routing matrices,
    local convolution, and approximate normalization. Softmax, top-k, gather,
    masks, and memory traffic are excluded; this is not a latency estimate.
    """
    if not any(isinstance(module, (BiLevelRoutingAttention, CBAM)) for module in network.modules()):
        return get_flops(network, imgsz)
    model = deepcopy(network).float().eval()
    reference = next(model.parameters())
    sample = torch.zeros(1, 3, imgsz, imgsz, device=reference.device)
    with torch.no_grad():
        macs, _ = thop.profile(model, inputs=(sample,), verbose=False,
                              custom_ops={BiLevelRoutingAttention: _bra_ops,
                                          CBAM: _cbam_ops,
                                          ChannelLayerNorm: _norm_ops,
                                          Attention: _attention_ops, AAttn: _attention_ops})
    return 2 * macs / 1e9
