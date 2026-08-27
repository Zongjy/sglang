#!/usr/bin/env python3
"""Small, shared description of a hybrid decoder layer layout.

The partition latency and memory models must agree on layer composition.  This
module is deliberately the only place that knows how a model config spells a
GDN/linear-attention layer.  The optimizer itself only sees ``gdn`` and
``full`` counts.

The first implementation supports the Qwen3.5-style ``layer_types`` field and
ordinary dense models (all layers are treated as full attention).  Unknown
hybrid layer names fail loudly instead of silently using a wrong estimate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class LayoutError(RuntimeError):
    """Raised when a model's layer layout cannot be determined safely."""


FULL_NAMES = frozenset(
    {"full", "full_attention", "attention", "sliding_attention", "swa"}
)
GDN_NAMES = frozenset(
    {"gdn", "linear", "linear_attention", "mamba", "linear_attn"}
)


def _normalise_kind(raw: object) -> str:
    name = str(raw).strip().lower()
    if name in FULL_NAMES:
        return "full"
    if name in GDN_NAMES:
        return "gdn"
    raise LayoutError(
        f"unsupported hybrid layer type {raw!r}; supported full names are "
        f"{sorted(FULL_NAMES)}, GDN names are {sorted(GDN_NAMES)}"
    )


@dataclass(frozen=True)
class LayerLayout:
    """Immutable layer-kind sequence plus O(1) range counts."""

    num_layers: int
    kinds: tuple[str, ...]
    prefix_gdn: tuple[int, ...] = ()
    prefix_full: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise LayoutError(f"num_layers must be positive, got {self.num_layers}")
        if len(self.kinds) != self.num_layers:
            raise LayoutError(
                f"layer kind count {len(self.kinds)} != num_layers {self.num_layers}"
            )
        if any(kind not in ("gdn", "full") for kind in self.kinds):
            raise LayoutError(f"invalid normalised layer kinds: {self.kinds!r}")
        if self.prefix_gdn and len(self.prefix_gdn) != self.num_layers + 1:
            raise LayoutError("prefix_gdn must have num_layers + 1 entries")
        if self.prefix_full and len(self.prefix_full) != self.num_layers + 1:
            raise LayoutError("prefix_full must have num_layers + 1 entries")
        if bool(self.prefix_gdn) != bool(self.prefix_full):
            raise LayoutError("prefix_gdn and prefix_full must be provided together")

    @classmethod
    def from_kinds(
        cls,
        kinds: list[str] | tuple[str, ...],
    ) -> "LayerLayout":
        normalised = tuple(_normalise_kind(kind) for kind in kinds)
        gdn = [0]
        full = [0]
        for kind in normalised:
            gdn.append(gdn[-1] + int(kind == "gdn"))
            full.append(full[-1] + int(kind == "full"))
        return cls(
            num_layers=len(normalised),
            kinds=normalised,
            prefix_gdn=tuple(gdn),
            prefix_full=tuple(full),
        )

    @classmethod
    def from_config(
        cls, config: Mapping[str, object]
    ) -> "LayerLayout":
        nested = config.get("text_config")
        text = nested if isinstance(nested, Mapping) else config
        raw_num_layers = text.get("num_hidden_layers")
        try:
            num_layers = int(raw_num_layers)
        except (TypeError, ValueError) as exc:
            raise LayoutError("config has no valid num_hidden_layers") from exc
        if num_layers <= 0:
            raise LayoutError(f"num_hidden_layers must be positive, got {num_layers}")

        raw_types = text.get("layer_types")
        if raw_types is not None:
            if not isinstance(raw_types, list) or len(raw_types) != num_layers:
                raise LayoutError(
                    "layer_types must be a list matching num_hidden_layers"
                )
            kinds = tuple(_normalise_kind(item) for item in raw_types)
        else:
            kinds = ("full",) * num_layers
        return cls.from_kinds(kinds)

    @classmethod
    def from_model_path(
        cls, model_path: str | Path, *, local_files_only: bool = True
    ) -> "LayerLayout":
        path = Path(model_path)
        if path.is_dir():
            config_path = path / "config.json"
            if not config_path.is_file():
                raise LayoutError(f"missing config.json under {path}")
            try:
                config = json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise LayoutError(f"cannot read model config {config_path}") from exc
            return cls.from_config(config)

        try:
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    str(model_path), local_files_only=local_files_only
                )
            )
        except Exception as exc:
            raise LayoutError(
                f"cannot resolve model config for {model_path!r}; "
                "provide a local model directory or an available HF cache entry"
            ) from exc
        config_path = snapshot / "config.json"
        if not config_path.is_file():
            raise LayoutError(f"cached model {model_path!r} has no config.json")
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise LayoutError(f"cannot read cached model config {config_path}") from exc
        return cls.from_config(config)

    def count_range(self, start: int, end: int) -> tuple[int, int]:
        """Return ``(gdn_count, full_count)`` for ``[start, end)``."""
        if not (0 <= start <= end <= self.num_layers):
            raise LayoutError(
                f"invalid layer range [{start}, {end}) for L={self.num_layers}"
            )
        if self.prefix_gdn:
            return (
                self.prefix_gdn[end] - self.prefix_gdn[start],
                self.prefix_full[end] - self.prefix_full[start],
            )
        segment = self.kinds[start:end]
        return segment.count("gdn"), segment.count("full")

def uniform_prefix_partition(
    num_layers: int, pp_size: int, l: int
) -> tuple[int, ...]:
    """Build the optimizer's ``[l, ..., l, residual]`` partition."""
    if pp_size <= 0:
        raise LayoutError(f"pp_size must be positive, got {pp_size}")
    if num_layers <= 0:
        raise LayoutError(f"num_layers must be positive, got {num_layers}")
    if pp_size == 1:
        if l not in (0, num_layers):
            raise LayoutError("PP=1 has no tunable boundary")
        return (num_layers,)
    if l <= 0 or (pp_size - 1) * l >= num_layers:
        raise LayoutError(
            f"l={l} does not leave a positive final stage for L={num_layers}, "
            f"P={pp_size}"
        )
    return (l,) * (pp_size - 1) + (num_layers - (pp_size - 1) * l,)
