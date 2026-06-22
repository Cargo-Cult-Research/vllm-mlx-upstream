# SPDX-License-Identifier: Apache-2.0
"""Unit tests for generic activation steering (vllm_mlx.steering).

Covers the contract that prevents the failure modes we hit in development:
  - INERT by default and at scale 0 (installing must not change normal serving),
  - correct per-layer magnitude (scale% * rms),
  - steer applies only at registered layers, only when active,
  - the active steer is cleared after the context (no bleed to the next generation),
  - registry load round-trips named vectors.
"""
import mlx.core as mx
import pytest

from vllm_mlx import steering


class _StubBlock:
    """Identity transformer block with the gpt-oss/mlx_lm __call__ signature."""
    def __call__(self, x, mask, cache=None):
        return x


class _StubModel:
    def __init__(self, n_layers, hidden):
        inner = type("Inner", (), {})()
        inner.layers = [_StubBlock() for _ in range(n_layers)]
        self.model = inner


@pytest.fixture
def model():
    # fresh install state per test
    steering._INSTALLED["done"] = False
    steering.clear_active()
    m = _StubModel(n_layers=6, hidden=4)
    steering.install(m)
    yield m
    steering.clear_active()


def _fwd(m, layer, x):
    return m.model.layers[layer](x, None)


def test_inert_by_default(model):
    x = mx.array([1.0, 2.0, 3.0, 4.0])
    assert mx.array_equal(_fwd(model, 2, x), x)


def test_inert_at_scale_zero(model):
    x = mx.array([1.0, 2.0, 3.0, 4.0])
    v = mx.array([10.0, 0.0, 0.0, 0.0])
    with steering.active({2: v}, scale=0.0, rms={2: 1.0}):
        assert mx.array_equal(_fwd(model, 2, x), x)


def test_applies_correct_magnitude(model):
    x = mx.zeros(4)
    v = mx.array([1.0, 0.0, 0.0, 0.0])
    # scale 50%, rms 8 -> magnitude 0.5*8 = 4.0
    with steering.active({3: v}, scale=50.0, rms={3: 8.0}):
        out = _fwd(model, 3, x)
    assert pytest.approx(float(out[0])) == 4.0
    assert pytest.approx(float(out[1])) == 0.0


def test_only_registered_layers(model):
    x = mx.zeros(4)
    v = mx.array([1.0, 1.0, 1.0, 1.0])
    with steering.active({3: v}, scale=100.0, rms={3: 1.0}):
        assert mx.array_equal(_fwd(model, 0, x), x)   # untouched layer
        assert not mx.array_equal(_fwd(model, 3, x), x)  # steered layer


def test_cleared_after_context_no_bleed(model):
    x = mx.zeros(4)
    v = mx.array([5.0, 0.0, 0.0, 0.0])
    with steering.active({1: v}, scale=100.0, rms={1: 1.0}):
        assert not mx.array_equal(_fwd(model, 1, x), x)
    # after the context, the next generation must be unaffected
    assert mx.array_equal(_fwd(model, 1, x), x)


def test_registry_roundtrip(tmp_path):
    f = tmp_path / "valence.safetensors"
    mx.save_safetensors(str(f), {"dir_8": mx.array([1.0, 0.0]),
                                 "rms_8": mx.array([2.0])})
    names = steering.load_registry(tmp_path)
    assert names == ["valence"]
    entry = steering.get("valence")
    assert 8 in entry["vectors"] and entry["rms"][8] == 2.0


def test_load_registry_missing_dir_is_safe(tmp_path):
    assert steering.load_registry(tmp_path / "nope") == []


# --- readout ---------------------------------------------------------------------

def test_hook_captures_mean_token_residual_when_active(model):
    x = mx.array([[[2.0, 4.0], [4.0, 8.0], [6.0, 0.0]]])   # (1, 3 tokens, hidden=2)
    steering._CAPTURE = {2: None}
    try:
        model.model.layers[2](x, None)
        assert steering._CAPTURE[2] is not None
        assert mx.allclose(steering._CAPTURE[2], mx.array([4.0, 4.0]))  # mean over tokens
        # a non-captured layer is untouched
        steering._CAPTURE = {2: None}
        model.model.layers[0](x, None)
        assert steering._CAPTURE[2] is None
    finally:
        steering._CAPTURE = None


def test_read_valence_math():
    steering._READ = {
        "band": [5],
        "base": {5: mx.zeros(3)},
        "pos": {5: mx.array([1.0, 0.0, 0.0])},
        "neg": {5: mx.array([0.0, 1.0, 0.0])},
        "psd": {5: 2.0},
        "nsd": {5: 4.0},
    }
    try:
        cap = {5: mx.array([6.0, 8.0, 0.0])}
        # pz = (6)/2 = 3 ; nz = (8)/4 = 2 ; valence = 1
        assert steering.read_valence(cap) == pytest.approx(1.0)
        assert steering.read_valence(None) is None
    finally:
        steering._READ = None


def test_read_band_and_valence_disabled_without_calib():
    steering._READ = None
    assert steering.read_band() == []
    assert steering.read_valence({5: mx.zeros(3)}) is None
