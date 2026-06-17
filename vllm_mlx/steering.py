# SPDX-License-Identifier: Apache-2.0
"""Generic inference-time activation steering for served MLX text models.

Adds steering vectors to the residual stream during generation (the technique
behind representation engineering / contrastive activation addition). The feature
is INERT by default: with no active steer — or scale 0 — the patched forward
returns the original output unchanged, so installing it cannot alter normal
serving or tool-call formatting.

Usage in an engine (serialized generation, e.g. SimpleEngine):

    steering.install(raw_model)          # once, after model load
    with steering.active(vectors, scale, rms):   # around one generation
        ... generate ...

`vectors` is {layer_index: unit_vector}; `scale` is a percentage; the per-layer
injection magnitude is ``scale/100 * rms[layer]`` so a single scale is comparable
across layers of differing residual norm. Steering vectors are model-specific
(a vector trained on one model won't transfer); load named vectors via
``load_registry`` and let requests select by name + scale.

Concurrency: the active steer is process-global, which is correct for engines that
serialize generation (one forward at a time). Batched/concurrent engines would need
per-sequence state — out of scope here (documented, not silently assumed).
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import mlx.core as mx

# Active steer for the current (serialized) generation, or None.
#   {"vectors": dict[int, mx.array], "scale": float, "rms": dict[int, float]}
_ACTIVE: dict | None = None
_INSTALLED: dict = {"done": False}
_REGISTRY: dict[str, dict] = {}


def set_active(vectors: dict, scale: float, rms: dict | None = None) -> None:
    global _ACTIVE
    _ACTIVE = {"vectors": vectors, "scale": float(scale), "rms": rms or {}}


def clear_active() -> None:
    global _ACTIVE
    _ACTIVE = None


@contextlib.contextmanager
def active(vectors: dict, scale: float, rms: dict | None = None):
    """Activate a steer for the duration of one (serialized) generation."""
    global _ACTIVE
    prev = _ACTIVE
    set_active(vectors, scale, rms)
    try:
        yield
    finally:
        _ACTIVE = prev


def install(raw_model) -> bool:
    """Patch the model's TransformerBlock.__call__ to add the active steer's
    contribution at each band layer. Idempotent; no-op when inactive/scale 0."""
    layers = raw_model.model.layers
    Block = type(layers[0])
    # Idempotent per class: never stack the hook (stacking would multiply the steer).
    if getattr(Block, "_steer_patched", False):
        for i, layer in enumerate(layers):
            layer._steer_idx = i
        _INSTALLED["done"] = True
        return True
    orig = Block.__call__

    def patched(self, x, mask, cache=None):
        out = orig(self, x, mask, cache)
        st = _ACTIVE
        if st is not None and st["scale"]:
            idx = getattr(self, "_steer_idx", -1)
            vec = st["vectors"].get(idx)
            if vec is not None:
                mag = st["scale"] / 100.0 * st["rms"].get(idx, 1.0)
                out = out + mag * vec
        return out

    Block.__call__ = patched
    Block._steer_patched = True
    for i, layer in enumerate(layers):
        layer._steer_idx = i
    _INSTALLED["done"] = True
    return True


def load_registry(path: str | Path) -> list[str]:
    """Load named steering vectors from a directory of safetensors files. Each file
    <name>.safetensors holds dir_<L> (unit vectors) and optional rms_<L>. Returns the
    names loaded. Safe to call with a missing dir (loads nothing)."""
    p = Path(path)
    _REGISTRY.clear()
    if not p.is_dir():
        return []
    for f in sorted(p.glob("*.safetensors")):
        d = mx.load(str(f))
        vectors, rms = {}, {}
        for k in d:
            if k.startswith("dir_"):
                vectors[int(k[4:])] = d[k]
            elif k.startswith("rms_"):
                rms[int(k[4:])] = float(d[k][0])
        if vectors:
            _REGISTRY[f.stem] = {"vectors": vectors, "rms": rms}
    return list(_REGISTRY)


def get(name: str) -> dict | None:
    return _REGISTRY.get(name)
