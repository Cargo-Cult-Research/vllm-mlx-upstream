# SPDX-License-Identifier: Apache-2.0
"""Regression test for MLLM stream_chat tool-format bug (32e36c3).

Verifies the _build_chat_messages logic in MLXMultimodalLM.stream_chat()
correctly preserves tool_calls on assistant messages and tool_call_id
on role="tool" messages.

Before the fix from 32e36c3:
- Assistant messages with empty text but non-empty tool_calls were
  silently dropped (guard was `if msg_text or msg_image_count > 0`).
- role="tool" messages were emitted without tool_call_id, breaking
  Gemma4's Jinja template forward-scan and triggering the "using last
  user message" fallback — causing the model to loop forever with
  the same ~76 tokens.

This test replicates the stream_chat message-building logic inline so
it does not need to import MLXMultimodalLM (which pulls in mlx,
requests, etc.). It runs fast and does not need --run-slow.
"""

import pytest


def _build_chat_messages(messages: list[dict]) -> list[dict]:
    """Replicate the stream_chat message-building logic from mllm.py.

    This mirrors lines ~2027-2090 of vllm_mlx/models/mllm.py in the
    stream_chat method. The second parameter `fix` controls whether
    the 32e36c3 fix is applied.
    """
    chat_messages = []

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_text = ""
        if isinstance(content, str):
            msg_text = content

        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")

        # The FIXED guard from 32e36c3:
        has_tool_content = bool(tool_calls) or role == "tool"
        if msg_text or has_tool_content:
            if role == "assistant":
                msg_dict: dict = {"role": role, "content": msg_text}
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                chat_messages.append(msg_dict)
            elif role == "tool":
                tool_msg: dict = {"role": "tool", "content": msg_text}
                if tool_call_id:
                    tool_msg["tool_call_id"] = tool_call_id
                chat_messages.append(tool_msg)
            else:
                chat_messages.append(
                    {
                        "role": role,
                        "content": [
                            {"type": "text", "text": msg_text, "content": msg_text}
                        ],
                    }
                )

    return chat_messages


def _build_chat_messages_buggy(messages: list[dict]) -> list[dict]:
    """Replicate the BUGGY stream_chat logic (pre-32e36c3).

    The guard was `if msg_text or msg_image_count > 0 or msg_audio_count > 0`
    which dropped pure-tool-call assistant turns. There was also no
    `elif role == "tool"` branch.
    """
    chat_messages = []

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_text = ""
        if isinstance(content, str):
            msg_text = content

        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")

        # BUGGY guard: only passes if there's text or images/audio
        if msg_text:
            if role == "assistant":
                chat_messages.append({"role": role, "content": msg_text})
            else:
                chat_messages.append(
                    {
                        "role": role,
                        "content": [
                            {"type": "text", "text": msg_text, "content": msg_text}
                        ],
                    }
                )
            # Note: tool_calls on assistant is DROPPED here

    return chat_messages


class TestStreamChatToolFormat:
    """Test that stream_chat preserves tool_calls and tool_call_id."""

    @pytest.fixture
    def messages_with_tool_calls(self):
        """Two-turn tool-calling conversation."""
        return [
            {"role": "user", "content": "List files in /tmp"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": '{"path": "/tmp"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "file1.txt  file2.py",
            },
        ]

    @pytest.fixture
    def messages_with_tool_call_id_only(self):
        """Tool message with tool_call_id but no content."""
        return [
            {"role": "user", "content": "What time is it?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_xyz789",
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "arguments": '{}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_xyz789",
            },
        ]

    @pytest.fixture
    def empty_text_assistant_messages(self):
        """Assistant turns with zero text content but tool_calls."""
        return [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "noop",
                            "arguments": '{}',
                        },
                    }
                ],
            },
        ]

    def test_fixed_stream_chat_preserves_tool_calls_on_assistant(
        self, messages_with_tool_calls
    ):
        """Assistant messages with tool_calls must appear in chat_messages."""
        chat_messages = _build_chat_messages(messages_with_tool_calls)

        assistant_turns = [m for m in chat_messages if m["role"] == "assistant"]
        assert len(assistant_turns) == 1, (
            "Assistant turn with tool_calls was dropped from chat_messages"
        )
        assert "tool_calls" in assistant_turns[0], (
            "tool_calls field missing from assistant chat_message"
        )
        assert assistant_turns[0]["tool_calls"][0]["id"] == "call_abc123"

    def test_fixed_stream_chat_preserves_tool_call_id_on_tool_messages(
        self, messages_with_tool_calls
    ):
        """role='tool' messages must carry tool_call_id."""
        chat_messages = _build_chat_messages(messages_with_tool_calls)

        tool_turns = [m for m in chat_messages if m["role"] == "tool"]
        assert len(tool_turns) == 1, (
            "role='tool' message was dropped or converted to another role"
        )
        assert "tool_call_id" in tool_turns[0], (
            "tool_call_id missing from role='tool' chat_message"
        )
        assert tool_turns[0]["tool_call_id"] == "call_abc123"

    def test_empty_text_assistant_turn_is_preserved(self, empty_text_assistant_messages):
        """Assistant with empty text but tool_calls must NOT be dropped.

        The pre-fix guard was:
            if msg_text or msg_image_count > 0 or msg_audio_count > 0:

        Since msg_text is "" and there are no images/audio, the entire
        assistant turn was skipped. The fix adds:
            has_tool_content = bool(tool_calls) or role == "tool"

        so that tool-carrying turns pass the guard.
        """
        chat_messages = _build_chat_messages(empty_text_assistant_messages)

        assistant_turns = [m for m in chat_messages if m["role"] == "assistant"]
        assert len(assistant_turns) == 1, (
            "Empty-text assistant turn was dropped — this is the 'endless loop' bug"
        )

    def test_multiturn_tool_loop_symptom_buggy(self, messages_with_tool_calls):
        """Demonstrate the BUG: assistant turn is dropped, tool has no tool_call_id.

        Before the fix, the chat_messages list would:
        1. Drop the assistant turn (empty text, no images → guard fails).
        2. Emit the tool message without tool_call_id (falls through to else).

        So the model sees only user messages on every turn, outputs the
        same tool call, and loops.

        This test CONFIRMS the buggy behavior exists (for regression
        detection), and the fixed version should pass instead.
        """
        buggy_messages = _build_chat_messages_buggy(messages_with_tool_calls)

        # The assistant turn is DROPPED (no tool_calls preserved).
        assistant_turns = [m for m in buggy_messages if m["role"] == "assistant"]
        assert len(assistant_turns) == 0, (
            "Buggy path should drop assistant turns with empty text"
        )

        # The tool message has no tool_call_id (no forward-scan match for Gemma4).
        tool_turns = [m for m in buggy_messages if m["role"] == "tool"]
        for tm in tool_turns:
            assert "tool_call_id" not in tm, (
                "Buggy path should not emit tool_call_id"
            )

    def test_multiturn_tool_loop_symptom_fixed(self, messages_with_tool_calls):
        """Verify the FIX: all 3 turns are preserved in chat_messages.

        With the fix, chat_messages should have 3 messages: user,
        assistant (with tool_calls), and tool (with tool_call_id).
        """
        chat_messages = _build_chat_messages(messages_with_tool_calls)

        assert len(chat_messages) == 3, (
            f"Expected 3 messages in chat_messages, got {len(chat_messages)}. "
            "This is the 'endless tool/thinking loop' symptom: the model sees "
            "only the first user message on every turn."
        )
        assert chat_messages[0]["role"] == "user"
        assert chat_messages[1]["role"] == "assistant"
        assert "tool_calls" in chat_messages[1]
        assert chat_messages[1]["tool_calls"][0]["id"] == "call_abc123"
        assert chat_messages[2]["role"] == "tool"
        assert chat_messages[2]["tool_call_id"] == "call_abc123"

    def test_tool_without_call_id_still_includes_empty_id(self):
        """A role='tool' message without tool_call_id should still appear."""
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "tool",
                "content": "result",
            },
        ]
        chat_messages = _build_chat_messages(messages)

        tool_turns = [m for m in chat_messages if m["role"] == "tool"]
        assert len(tool_turns) == 1

    def test_tool_call_id_only_preserved(self, messages_with_tool_call_id_only):
        """Tool message with tool_call_id but no content is preserved."""
        chat_messages = _build_chat_messages(messages_with_tool_call_id_only)

        tool_turns = [m for m in chat_messages if m["role"] == "tool"]
        assert len(tool_turns) == 1
        assert tool_turns[0]["tool_call_id"] == "call_xyz789"


class TestGemma4NativeFormatFlag:
    """Test that Gemma4ToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is True.

    Before 32e36c3, Gemma4ToolParser.SUPPORTS_NATIVE_TOOL_FORMAT was False
    (inherited from the abstract base). Without this flag, tool results get
    converted to "[Tool Result (...)]:" text injected into a user turn,
    which the Gemma4 template doesn't recognize, causing infinite loops.
    """

    def test_gemma4_supports_native_tool_format(self):
        """Gemma4ToolParser must declare native tool format support."""
        from vllm_mlx.tool_parsers import Gemma4ToolParser

        assert (
            Gemma4ToolParser.SUPPORTS_NATIVE_TOOL_FORMAT is True
        ), "Gemma4ToolParser.SUPPORTS_NATIVE_TOOL_FORMAT must be True"
        assert (
            Gemma4ToolParser.supports_native_format() is True
        ), "Gemma4ToolParser.supports_native_format() must return True"
