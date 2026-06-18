# SPDX-License-Identifier: Apache-2.0
"""Regression tests: per-request activation steering must be ACTIVE during generation
on BOTH the non-streaming and streaming chat paths.

This pins the bug found in development: engine.chat routes a non-mllm / no-tools
request through self._model.chat *directly* (not through stream_chat), so wrapping
only stream_chat left that path unsteered. These tests fail if either path skips the
steer or if it leaks past the request (no reset).

No MLX ops here — the seam only sets a small state dict, which the mock records — so
these are immune to the gpu-stream/test-ordering flakiness in this environment.
"""
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx import steering
from vllm_mlx.engine.simple import SimpleEngine

SPEC = {"vectors": {8: object()}, "scale": 4.0, "rms": {8: 1.0}}


@pytest.fixture
def mock_llm_model():
    m = MagicMock()
    m.tokenizer = MagicMock()
    m.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    m.tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3])
    return m


@pytest.fixture(autouse=True)
def _clean_steer():
    steering.clear_active()
    yield
    steering.clear_active()


@pytest.mark.anyio
async def test_nonstream_path_applies_steering(mock_llm_model):
    """REGRESSION: non-mllm/no-tools chat goes through self._model.chat directly."""
    seen = {}

    def chat_side_effect(**kwargs):
        seen["active"] = steering._ACTIVE   # snapshot the steer state during generation
        out = MagicMock(); out.text = "ok"; out.tokens = [1]; out.finish_reason = "stop"
        return out

    mock_llm_model.chat = MagicMock(side_effect=chat_side_effect)
    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("test-model")
        engine._model = mock_llm_model
        engine._loaded = True
        await engine.chat([{"role": "user", "content": "hi"}], tools=None, steering=SPEC)

    mock_llm_model.chat.assert_called_once()
    assert seen["active"] is not None, "steer was NOT active during non-stream generation"
    assert seen["active"]["scale"] == 4.0
    assert steering._ACTIVE is None, "steer leaked past the request (not reset)"


@pytest.mark.anyio
async def test_stream_path_applies_steering(mock_llm_model):
    """Streaming chat must also be steered (wrapper around _stream_chat_impl)."""
    seen = {}

    async def fake_impl(*args, **kwargs):
        seen["active"] = steering._ACTIVE
        out = MagicMock(); out.text = "ok"; out.tokens = [1]; out.finish_reason = "stop"
        yield out

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("test-model")
        engine._model = mock_llm_model
        engine._loaded = True
        engine._stream_chat_impl = fake_impl  # type: ignore[method-assign]
        async for _ in engine.stream_chat([{"role": "user", "content": "hi"}], steering=SPEC):
            pass

    assert seen["active"] is not None, "steer was NOT active during streaming generation"
    assert seen["active"]["scale"] == 4.0
    assert steering._ACTIVE is None, "steer leaked past the request (not reset)"


@pytest.mark.anyio
async def test_no_spec_is_inert(mock_llm_model):
    """No steering kwarg → steer stays inactive (normal serving unaffected)."""
    seen = {}

    def chat_side_effect(**kwargs):
        seen["active"] = steering._ACTIVE
        out = MagicMock(); out.text = "ok"; out.tokens = [1]; out.finish_reason = "stop"
        return out

    mock_llm_model.chat = MagicMock(side_effect=chat_side_effect)
    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=False):
        engine = SimpleEngine("test-model")
        engine._model = mock_llm_model
        engine._loaded = True
        await engine.chat([{"role": "user", "content": "hi"}], tools=None)

    assert seen["active"] is None, "steer active without a steering spec"
