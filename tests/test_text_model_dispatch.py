"""Guard: build_text_model must only build an mlx_lm TextModel for
architectures that actually have one. Unknown types (e.g. mistral3 / Pixtral)
return None so the engine uses the MLLM stream_chat path, rather than building
a structurally-wrong qwen3_5 model from foreign weights (which loaded fine but
then crashed the text-route decode and would have produced garbage).
"""

from vllm_mlx.text_model_from_vlm import (
    _import_text_model_classes,
    build_text_model,
)


def test_gemma4_maps_to_a_class():
    Model, Args = _import_text_model_classes("gemma4_text")
    assert Model is not None and Args is not None


def test_qwen3_5_maps_to_a_class():
    for mt in ("qwen3_5_text", "qwen3_5"):
        Model, Args = _import_text_model_classes(mt)
        assert Model is not None and Args is not None


def test_mistral3_has_no_text_model():
    Model, Args = _import_text_model_classes("mistral3")
    assert Model is None and Args is None


def test_unknown_type_has_no_text_model():
    Model, Args = _import_text_model_classes("pixtral")
    assert Model is None and Args is None


def test_build_text_model_none_for_unsupported(tmp_path):
    """build_text_model returns None (not a wrong model) for mistral3."""
    (tmp_path / "config.json").write_text('{"model_type": "mistral3"}')

    class _FakeVLM:
        language_model = object()

    assert build_text_model(_FakeVLM(), tmp_path) is None
