# SPDX-License-Identifier: Apache-2.0
"""Bind mlx-lm/mlx-vlm generation streams to the engine's worker thread.

MLX streams are per-thread runtime state: an array tagged to a stream on
thread A cannot be read from thread B (it raises "no Stream(gpu, N) in
current thread"). SimpleEngine pins all blocking MLX work to a single
dedicated worker thread, so we only need ONE generation stream for the
lifetime of the process — created the first time this is called from the
worker, then reused.

mlx_lm and mlx_vlm both define module-level ``generation_stream`` at import
time on whichever thread loads them (typically the main thread). Those
imports may bind to a stream the MLX worker thread cannot resolve, so we
overwrite both module attributes with our worker-owned stream.
"""

import importlib
import threading
from collections.abc import Iterable

import mlx.core as mx

_LOCK = threading.Lock()
_GENERATION_STREAM = None


def bind_generation_streams(
    module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"),
) -> object:
    """Idempotently bind mlx-lm/mlx-vlm generation streams to this thread.

    First call (from the pinned MLX worker thread) creates a fresh stream
    and stashes it. Subsequent calls reuse the same stream — no new streams
    are ever created. This avoids the "Stream(gpu, N) GC / cross-thread"
    crash class entirely.

    Must only be called from the engine's pinned worker thread. Calling
    from another thread will succeed but pin the global stream to the
    first caller's thread, which is almost never what you want.
    """
    global _GENERATION_STREAM
    with _LOCK:
        if _GENERATION_STREAM is None:
            _GENERATION_STREAM = mx.new_stream(mx.default_device())
        mx.set_default_stream(_GENERATION_STREAM)
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            if hasattr(module, "generation_stream"):
                setattr(module, "generation_stream", _GENERATION_STREAM)
        return _GENERATION_STREAM
