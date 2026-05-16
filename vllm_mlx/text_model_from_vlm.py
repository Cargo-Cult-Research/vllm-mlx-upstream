# SPDX-License-Identifier: Apache-2.0
"""Construct an mlx_lm TextModel from mlx_vlm-loaded model weights.

When mlx_vlm loads a model, it strips MTP weights in sanitize().
This module builds a parallel mlx_lm TextModel that:
1. Shares backbone + lm_head weights with the vlm model (zero-copy)
2. Loads MTP weights from safetensors on disk
3. Provides full mlx_lm API: return_hidden, n_confirmed, mtp_forward, make_mtp_cache
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

logger = logging.getLogger(__name__)


def build_text_model(vlm_model: Any, model_path: str | Path) -> Any | None:
    """Build an mlx_lm text Model from a vlm-loaded model's weights.

    Dispatches on text_config['model_type']:
      - qwen3_5_text  -> mlx_lm.models.qwen3_5 (with MTP injection)
      - gemma4_text   -> mlx_lm.models.gemma4_text (no MTP)
      - anything else -> None (caller falls back to the MLLM path)

    Args:
        vlm_model: The mlx_vlm-loaded model (has .language_model attribute)
        model_path: Path to the model directory (contains config.json + safetensors)

    Returns:
        An mlx_lm Model sharing weights (zero-copy) with vlm_model.language_model,
        or None if the architecture isn't supported or the build failed.
    """
    if vlm_model is None:
        return None

    model_path = Path(model_path) if model_path else None
    if model_path is None or not (model_path / "config.json").exists():
        return None

    try:
        config = json.loads((model_path / "config.json").read_text())
        text_config = config.get("text_config", config)
        text_model_type = text_config.get("model_type", "")

        if text_model_type == "qwen3_5_text":
            return _build_qwen3_5(vlm_model, model_path, config, text_config)
        if text_model_type == "gemma4_text":
            return _build_gemma4(vlm_model, model_path, config, text_config)

        logger.info(
            "No mlx_lm text-model dispatch for model_type=%r; "
            "text-only requests will use the MLLM path",
            text_model_type,
        )
        return None

    except ImportError as e:
        logger.error("Cannot import mlx_lm text model module: %s", e)
        return None
    except Exception as e:
        logger.error("Failed to build TextModel from vlm: %s", e)
        logger.debug("build_text_model traceback:\n%s", traceback.format_exc())
        return None


def _quantize_to_source(
    text_model: Any,
    text_config: dict,
    config: dict,
    all_weight_names: set[str],
) -> None:
    """Quantize text_model layers to match the source weights on disk.

    - Skips modules whose quantized weights aren't present on disk (looked up by
      `<path>.scales` in `all_weight_names`), so BF16 layers (e.g. MTP fc) stay
      unquantized.
    - Honors per-layer overrides in the quantization config (e.g. Gemma 4 stores
      MLP/router projections at 8-bit while the rest of the model is 4-bit).
    """
    quantization = text_config.get("quantization", config.get("quantization", None))
    if quantization is None:
        return

    default_group_size = quantization.get("group_size", 64)
    default_bits = quantization.get("bits", 8)

    # Build path -> {group_size, bits} from per-layer overrides. Keys on disk may
    # be prefixed "language_model." (mlx_vlm storage) while the built text model
    # uses bare "model.*" paths — index under both forms so either matches.
    overrides: dict[str, dict] = {}
    for key, val in quantization.items():
        if not isinstance(val, dict):
            continue
        overrides[key] = val
        if key.startswith("language_model."):
            overrides[key[len("language_model.") :]] = val

    def _class_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if f"{path}.scales" not in all_weight_names:
            return False
        if path in overrides:
            o = overrides[path]
            return {
                "group_size": o.get("group_size", default_group_size),
                "bits": o.get("bits", default_bits),
            }
        return True

    nn.quantize(
        text_model,
        group_size=default_group_size,
        bits=default_bits,
        class_predicate=_class_predicate,
    )


def _build_qwen3_5(
    vlm_model: Any, model_path: Path, config: dict, text_config: dict
) -> Any | None:
    # qwen3_5.TextModel/TextModelArgs handle both dense and MoE natively
    # (MTPDecoderLayer auto-selects SparseMoeBlock when args.num_experts > 0).
    # qwen3_5_moe.py does NOT export these.
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    args = TextModelArgs.from_dict(text_config)
    text_model = TextModel(args)

    vlm_lm = vlm_model.language_model
    vlm_weights = mlx.utils.tree_flatten(vlm_lm.parameters())
    mtp_weights = _load_mtp_weights(model_path)

    all_weight_names = set(name for name, _ in vlm_weights)
    all_weight_names.update(name for name, _ in mtp_weights)

    _quantize_to_source(text_model, text_config, config, all_weight_names)

    # strict=False because TextModel has MTP params that vlm doesn't have yet.
    text_model.load_weights(vlm_weights, strict=False)
    logger.info(
        "Transferred %d weight arrays from vlm language_model", len(vlm_weights)
    )

    if mtp_weights:
        text_model.load_weights(mtp_weights, strict=False)
        logger.info("Loaded %d MTP weights from safetensors", len(mtp_weights))
    else:
        logger.warning("No MTP weights found in %s", model_path.name)

    # Inject MTP if TextModel doesn't have native MTP support.
    # mlx_lm's qwen3_5.TextModel strips MTP weights in sanitize(),
    # so we inject MTP module + methods at runtime.
    if not hasattr(text_model, "mtp") or text_model.mtp is None:
        num_mtp = text_config.get("mtp_num_hidden_layers", 0)
        if num_mtp == 0:
            num_mtp = text_config.get("num_nextn_predict_layers", 0)
        if num_mtp > 0:
            from .patches.qwen3_5_mtp import inject_mtp_support

            inject_mtp_support(text_model, model_path, config)

    if hasattr(text_model, "mtp") and text_model.mtp is not None:
        mx.eval(text_model.mtp.parameters())
        num_mtp = text_config.get(
            "mtp_num_hidden_layers",
            text_config.get("num_nextn_predict_layers", 0),
        )
        logger.info("TextModel built with MTP support (%d layers)", num_mtp)
    else:
        logger.info("TextModel built without MTP")

    return text_model


def _build_gemma4(
    vlm_model: Any, model_path: Path, config: dict, text_config: dict
) -> Any | None:
    # Gemma 4 is a different MoE topology (top_k_experts, per-layer inputs,
    # K-eq-V global attention, KV-shared layers) and has no MTP. mlx_lm ships
    # a dedicated gemma4_text module whose Gemma4TextModel mirrors the mlx_vlm
    # submodule layout (model.embed_tokens / layers / norm; Experts.switch_glu),
    # so weights from vlm_model.language_model load directly.
    from mlx_lm.models.gemma4_text import Model as Gemma4Model
    from mlx_lm.models.gemma4_text import ModelArgs as Gemma4ModelArgs

    args = Gemma4ModelArgs.from_dict(text_config)
    text_model = Gemma4Model(args)

    vlm_lm = vlm_model.language_model
    vlm_weights = mlx.utils.tree_flatten(vlm_lm.parameters())
    all_weight_names = set(name for name, _ in vlm_weights)

    _quantize_to_source(text_model, text_config, config, all_weight_names)

    # Tied word embeddings on Gemma 4 -> no lm_head on either side, so
    # strict=False just absorbs any incidental mismatches without surprises.
    text_model.load_weights(vlm_weights, strict=False)
    logger.info(
        "Transferred %d weight arrays from vlm language_model (gemma4)",
        len(vlm_weights),
    )
    logger.info("TextModel built (gemma4_text, no MTP)")

    return text_model


def _load_mtp_weights(model_path: Path) -> list[tuple[str, mx.array]]:
    """Load MTP weights from safetensors, stripping the language_model. prefix.

    mlx_vlm's sanitize() strips mtp.* keys during model loading,
    but the weights are still on disk in the safetensors files.
    """
    index_file = model_path / "model.safetensors.index.json"
    if not index_file.exists():
        return []

    index = json.loads(index_file.read_text())
    weight_map = index.get("weight_map", {})

    # Find MTP keys and their shard files
    mtp_keys: dict[str, tuple[str, str]] = {}
    for key, shard in weight_map.items():
        if ".mtp." in key:
            # Strip "language_model." prefix to match mlx_lm namespace
            clean = (
                key.replace("language_model.", "", 1)
                if key.startswith("language_model.")
                else key
            )
            mtp_keys[key] = (clean, shard)

    if not mtp_keys:
        return []

    # Group by shard to minimize I/O
    shards: dict[str, list[tuple[str, str]]] = {}
    for orig, (clean, shard) in mtp_keys.items():
        shards.setdefault(shard, []).append((orig, clean))

    weights = []
    for shard_file, key_pairs in shards.items():
        shard_path = model_path / shard_file
        if not shard_path.exists():
            logger.warning("MTP shard not found: %s", shard_file)
            continue
        shard_data = mx.load(str(shard_path))
        for orig, clean in key_pairs:
            if orig in shard_data:
                weights.append((clean, shard_data[orig]))

    return weights
