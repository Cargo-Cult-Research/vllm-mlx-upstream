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
# Readout: capture residuals at the read band during a forward, then project onto a
# loaded read calibration to get a perceived-valence scalar. _CAPTURE is {layer: mean-
# token residual} while a capture is in flight, else None. _READ holds the calibration.
_CAPTURE: dict | None = None
_READ: dict | None = None


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
        idx = getattr(self, "_steer_idx", -1)
        st = _ACTIVE
        if st is not None and st["scale"]:
            vec = st["vectors"].get(idx)
            if vec is not None:
                mag = st["scale"] / 100.0 * st["rms"].get(idx, 1.0)
                out = out + mag * vec
        if _CAPTURE is not None and idx in _CAPTURE:
            _CAPTURE[idx] = mx.mean(out[0].astype(mx.float32), axis=0)  # (hidden,)
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


# --- Readout (perceived valence from activations) -------------------------------

def load_read_calib(path: str | Path) -> list[int]:
    """Load a neutral-anchored two-pole read calibration: per band layer, base
    (neutral mean), pos/neg unit dirs, and psd/nsd projection scales. Returns the
    band. Safe with a missing file (readout stays disabled)."""
    global _READ
    p = Path(path)
    if not p.is_file():
        return []
    d = mx.load(str(p))
    band = sorted(int(k[5:]) for k in d if k.startswith("base_"))
    R: dict = {"band": band, "base": {}, "pos": {}, "neg": {}, "psd": {}, "nsd": {}}
    for L in band:
        R["base"][L] = d[f"base_{L}"]; R["pos"][L] = d[f"pos_{L}"]; R["neg"][L] = d[f"neg_{L}"]
        R["psd"][L] = float(d[f"psd_{L}"][0]); R["nsd"][L] = float(d[f"nsd_{L}"][0])
    _READ = R
    return band


def read_band() -> list[int]:
    return list(_READ["band"]) if _READ else []


def run_capture(raw_model, ids: list[int]) -> dict | None:
    """Forward `ids` through the model capturing mean-token residuals at the read
    band. MUST run on the MLX worker thread (streams bound). Returns {layer: vec}."""
    global _CAPTURE
    if _READ is None:
        return None
    _CAPTURE = {L: None for L in _READ["band"]}
    try:
        raw_model.model(mx.array([ids]))
        mx.eval([v for v in _CAPTURE.values() if v is not None])
        return dict(_CAPTURE)
    finally:
        _CAPTURE = None


def read_valence(captured: dict | None) -> float | None:
    """Perceived valence = mean over the band of (positivity_z - negativity_z),
    neutral-anchored. Same math as the offline read meter. None if uncalibrated."""
    if _READ is None or not captured:
        return None
    R = _READ
    vals = []
    for L in R["band"]:
        f = captured.get(L)
        if f is None:
            continue
        pz = float((f - R["base"][L]) @ R["pos"][L]) / R["psd"][L]
        nz = float((f - R["base"][L]) @ R["neg"][L]) / R["nsd"][L]
        vals.append(pz - nz)
    return sum(vals) / len(vals) if vals else None
