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
from typing import Any, Mapping, Sequence


class LayoutError(RuntimeError):
    """Raised when a model's layer layout cannot be determined safely."""


FULL_NAMES = frozenset(
    {"full", "full_attention", "attention", "sliding_attention", "swa"}
)
GDN_NAMES = frozenset(
    {"gdn", "linear", "linear_attention", "mamba", "linear_attn"}
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    # transformers configs expose attributes rather than Mapping methods.
    result: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:  # pragma: no cover - unusual custom config object
            continue
        if isinstance(item, (str, int, float, bool, list, tuple, dict)):
            result[name] = item
    return result


def _text_config(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping) and hasattr(config, "text_config"):
        nested = getattr(config, "text_config")
        if nested is not None:
            return _as_mapping(nested)
    mapping = _as_mapping(config)
    nested = mapping.get("text_config")
    return _as_mapping(nested) if nested is not None else mapping


def _normalise_kind(raw: Any) -> str:
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
    model_type: str = ""
    source: str = "config"
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
        kinds: Sequence[str],
        *,
        model_type: str = "",
        source: str = "config",
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
            model_type=model_type,
            source=source,
            prefix_gdn=tuple(gdn),
            prefix_full=tuple(full),
        )

    @classmethod
    def from_config(
        cls, config: Any, *, source: str = "config"
    ) -> "LayerLayout":
        text = _text_config(config)
        raw_num_layers = text.get("num_hidden_layers")
        try:
            num_layers = int(raw_num_layers)
        except (TypeError, ValueError) as exc:
            raise LayoutError("config has no valid num_hidden_layers") from exc
        if num_layers <= 0:
            raise LayoutError(f"num_hidden_layers must be positive, got {num_layers}")

        raw_types = text.get("layer_types")
        if raw_types:
            if isinstance(raw_types, (str, bytes)):
                raise LayoutError("layer_types must be a sequence, not a string")
            if len(raw_types) != num_layers:
                raise LayoutError(
                    f"layer_types length {len(raw_types)} != num_hidden_layers "
                    f"{num_layers}"
                )
            kinds = tuple(_normalise_kind(item) for item in raw_types)
        else:
            # Dense models do not carry layer_types.  For Qwen3.5-like configs
            # that omit the explicit list, reconstruct the documented periodic
            # layout from full_attention_interval.  We only use this fallback
            # when linear-attention fields are present; otherwise all layers
            # are ordinary full-attention layers.
            interval = text.get("full_attention_interval")
            has_linear_fields = any(
                key.startswith("linear_") or key.startswith("mamba_")
                for key in text
            )
            if interval is not None and has_linear_fields:
                try:
                    interval = int(interval)
                except (TypeError, ValueError) as exc:
                    raise LayoutError(
                        f"invalid full_attention_interval={interval!r}"
                    ) from exc
                if interval <= 0:
                    raise LayoutError("full_attention_interval must be positive")
                kinds = tuple(
                    "full" if (index + 1) % interval == 0 else "gdn"
                    for index in range(num_layers)
                )
            else:
                kinds = ("full",) * num_layers

        model_type = str(text.get("model_type", ""))
        return cls.from_kinds(kinds, model_type=model_type, source=source)

    @classmethod
    def from_model_path(
        cls, model_path: str | Path, *, local_files_only: bool = True
    ) -> "LayerLayout":
        path = Path(model_path)
        source = str(path)
        if path.is_file():
            try:
                return cls.from_config(json.loads(path.read_text()), source=source)
            except (OSError, json.JSONDecodeError) as exc:
                raise LayoutError(f"cannot read model config {path}") from exc
        if path.is_dir():
            config_path = path / "config.json"
            if not config_path.is_file():
                raise LayoutError(f"missing config.json under {path}")
            try:
                config = json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise LayoutError(f"cannot read model config {config_path}") from exc
            return cls.from_config(config, source=source)

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
        return cls.from_config(config, source=str(config_path))

    @property
    def gdn_total(self) -> int:
        return self.prefix_gdn[-1] if self.prefix_gdn else self.kinds.count("gdn")

    @property
    def full_total(self) -> int:
        return self.prefix_full[-1] if self.prefix_full else self.kinds.count("full")

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

    def ranges_for_partition(
        self, partition: Sequence[int]
    ) -> tuple[tuple[int, int], ...]:
        counts = tuple(int(value) for value in partition)
        if not counts or any(value <= 0 for value in counts):
            raise LayoutError(f"partition must contain positive counts: {partition!r}")
        if sum(counts) != self.num_layers:
            raise LayoutError(
                f"partition sums to {sum(counts)}, expected {self.num_layers}"
            )
        ranges = []
        start = 0
        for count in counts:
            ranges.append((start, start + count))
            start += count
        return tuple(ranges)

    def composition(self, partition: Sequence[int]) -> tuple[str, ...]:
        return tuple(
            f"{gdn}G+{full}F"
            for gdn, full in (
                self.count_range(start, end)
                for start, end in self.ranges_for_partition(partition)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "model_type": self.model_type,
            "source": self.source,
            "layer_types": list(self.kinds),
            "gdn_total": self.gdn_total,
            "full_total": self.full_total,
        }


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
