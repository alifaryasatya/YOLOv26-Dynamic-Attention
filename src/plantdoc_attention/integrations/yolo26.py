"""YOLO26 P4 integration preserving layer indices and pretrained weights."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

from torch import Tensor, nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C3k2, Detect

from ..modules import ResidualAttention, ResidualCBAM


@dataclass(frozen=True)
class BRAConfig:
    layer: int = 19
    dim: int = 128
    qk_dim: int = 32
    num_heads: int = 4
    n_win: int = 4
    topk: int = 4
    side_dwconv: int = 3
    attn_drop: float = 0.0
    gamma_init: float = 0.0


@dataclass(frozen=True)
class CBAMConfig:
    kind: str = "cbam"
    layer: int = 19
    dim: int = 128
    reduction_ratio: int = 16
    spatial_kernel: int = 7
    gamma_init: float = 0.0


def parse_attention_config(config=None) -> BRAConfig | CBAMConfig:
    if isinstance(config, (BRAConfig, CBAMConfig)):
        if isinstance(config, CBAMConfig) and config.kind != "cbam":
            raise ValueError("CBAMConfig.kind must be cbam")
        return config
    values = dict(config or {})
    kind = values.pop("kind", "bra")
    if kind == "bra":
        return BRAConfig(**values)
    if kind == "cbam":
        return CBAMConfig(**values)
    raise ValueError(f"Unknown attention kind: {kind}")


class P4AttentionWrapper(nn.Module):
    """Execute original P4 layer, then residual attention, retaining graph metadata."""

    def __init__(self, original: nn.Module, config: BRAConfig | CBAMConfig) -> None:
        super().__init__()
        self.original = original
        kwargs = asdict(config)
        kwargs.pop("layer")
        kwargs.pop("kind", None)
        self.residual = ResidualCBAM(**kwargs) if isinstance(config, CBAMConfig) else ResidualAttention(**kwargs)
        self.config = asdict(config)
        self.i, self.f = original.i, deepcopy(original.f)
        self.type = "plantdoc_attention.P4AttentionWrapper"
        self.np = sum(parameter.numel() for parameter in self.parameters())

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(self.original(x))


def attach_bra_p4(network: nn.Module, config: BRAConfig | dict | None = None) -> nn.Module:
    config = parse_attention_config(config)
    if not isinstance(config, BRAConfig):
        raise ValueError("attach_bra_p4 requires a BRA config")
    return attach_attention_p4(network, config)


def attach_attention_p4(network: nn.Module, config=None) -> nn.Module:
    """Mutate a loaded DetectionModel once; explicitly reject incompatible graphs."""
    config = parse_attention_config(config)
    if config.layer != 19:
        raise ValueError("Current integration supports only YOLO26 P4 layer 19")
    if not hasattr(network, "model") or len(network.model) != 24:
        raise ValueError("Expected the verified 24-layer YOLO26 detection graph")
    layer = network.model[config.layer]
    if isinstance(layer, P4AttentionWrapper):
        raise ValueError("P4 already contains attention; refusing duplicate wrapping")
    head = network.model[-1]
    # `head.end2end` is toggled by inference/NMS settings on checkpoint load;
    # validate architecture metadata instead of that mutable runtime flag.
    if (not isinstance(layer, C3k2) or not isinstance(head, Detect)
            or not network.yaml.get("end2end") or head.reg_max != 1):
        raise ValueError("Expected YOLO26 C3k2 P4 and end-to-end Detect head")
    if layer.i != 19 or layer.f != -1 or head.f != [16, 19, 22] or 19 not in network.save:
        raise ValueError("YOLO graph metadata does not match the verified P4 topology")
    channels = layer.cv2.conv.out_channels
    if channels != config.dim:
        raise ValueError(f"P4 outputs {channels} channels; config requests {config.dim}")
    wrapper = P4AttentionWrapper(layer, config)
    reference = next(layer.parameters())
    wrapper.residual.to(device=reference.device, dtype=reference.dtype)
    wrapper.train(layer.training)
    network.model[config.layer] = wrapper
    network.yaml = deepcopy(network.yaml)
    network.yaml["plantdoc_attention"] = asdict(config)
    return network


class AttentionDetectionTrainer(DetectionTrainer):
    """Rebuild the base graph, transfer original weights, and reattach P4 BRA.

    Attention config is passed through the constructor or stored model YAML,
    never as unknown Ultralytics train arguments. Supports wrapped checkpoints
    and class-count changes through the normal detection transfer path.
    """

    def __init__(self, *args, attention_config: dict | BRAConfig | None = None, **kwargs) -> None:
        self.attention_config = attention_config
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        stored_config = cfg.get("plantdoc_attention") if isinstance(cfg, dict) else None
        selected_config = self.attention_config or stored_config
        if selected_config is None:
            raise ValueError("Attention trainer requires attention_config or checkpoint YAML metadata")
        config = parse_attention_config(selected_config)
        wrapped_source = None
        if isinstance(weights, nn.Module) and isinstance(weights.model[config.layer], P4AttentionWrapper):
            wrapped_source = weights.model[config.layer]
            if asdict(parse_attention_config(wrapped_source.config)) != asdict(config):
                raise ValueError("Checkpoint attention config differs from requested config")
            # Undo only the wrapper in a copy so pretrained original keys match.
            base_weights = deepcopy(weights)
            base_weights.model[config.layer] = base_weights.model[config.layer].original
        else:
            base_weights = weights
        clean_cfg = deepcopy(cfg)
        if isinstance(clean_cfg, dict):
            clean_cfg.pop("plantdoc_attention", None)
        model = super().get_model(cfg=clean_cfg, weights=base_weights, verbose=verbose)
        attach_attention_p4(model, config)
        if wrapped_source is not None:
            model.model[config.layer].residual.load_state_dict(wrapped_source.residual.state_dict(), strict=True)
        return model


# Preserve the Phase 2/3 import for scripts and historical workflows.
BRADetectionTrainer = AttentionDetectionTrainer
