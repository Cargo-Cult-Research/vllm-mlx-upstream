"""Tests for folding a system role into the first user turn.

Several multimodal chat templates (Mistral 3 / Pixtral, Gemma 4) reject a
`system` role. The failure path in mllm.py folds system into the first user
message and retries, instead of collapsing to a bare last-user-message (which
used to drop the tool definitions and make tool calls leak as plain text).
"""

from vllm_mlx.models.mllm import (
    _content_text,
    _fold_system_into_first_user,
    _last_user_text,
)


def test_no_system_returns_none():
    msgs = [{"role": "user", "content": "hi"}]
    assert _fold_system_into_first_user(msgs) is None


def test_fold_into_string_user_content():
    msgs = [
        {"role": "system", "content": "You are Claude Code."},
        {"role": "user", "content": "List /tmp."},
    ]
    folded = _fold_system_into_first_user(msgs)
    assert folded is not None
    assert all(m["role"] != "system" for m in folded)
    assert folded[0]["role"] == "user"
    assert "You are Claude Code." in folded[0]["content"]
    assert "List /tmp." in folded[0]["content"]


def test_fold_into_list_user_content_preserves_blocks():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": [{"type": "text", "text": "do it"}]},
    ]
    folded = _fold_system_into_first_user(msgs)
    assert folded[0]["role"] == "user"
    assert isinstance(folded[0]["content"], list)
    # system text is prepended as its own text block, original block kept
    assert folded[0]["content"][0]["text"] == "SYS"
    assert folded[0]["content"][-1]["text"] == "do it"


def test_fold_with_no_user_promotes_system_to_user():
    msgs = [{"role": "system", "content": "SYS"}]
    folded = _fold_system_into_first_user(msgs)
    assert len(folded) == 1
    assert folded[0]["role"] == "user"
    assert _content_text(folded[0]["content"]) == "SYS"


def test_fold_does_not_mutate_original():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u"},
    ]
    _fold_system_into_first_user(msgs)
    assert msgs[0]["role"] == "system"  # original untouched
    assert msgs[1]["content"] == "u"


def test_last_user_text_handles_list_and_str():
    assert _last_user_text([{"role": "user", "content": "x"}]) == "x"
    assert (
        _last_user_text([{"role": "user", "content": [{"type": "text", "text": "y"}]}])
        == "y"
    )
    assert _last_user_text([{"role": "assistant", "content": "a"}]) == ""
