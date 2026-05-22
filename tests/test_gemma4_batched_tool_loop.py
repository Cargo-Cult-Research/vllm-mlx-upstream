# SPDX-License-Identifier: Apache-2.0
"""Integration test: Gemma 4 + continuous batching must not degrade on
long tool-result prompts.

Bisection (2026-05-22):
- SimpleEngine path (`vllm-mlx serve ...` without --continuous-batching):
  WORKS — returns a coherent answer to a long tool-call follow-up.
- BatchedEngine path (`--continuous-batching`, any --max-num-seqs):
  BROKEN — model emits degenerate token loops drawn from prompt tokens,
  e.g. ``mtp_en-mtp_en-mtp_en-...``. ``finish_reason`` ends as ``length``.

Confirmed independent of:
- --kv-cache-quantization (loops with or without)
- --enable-prefix-cache (loops with or without)
- --mllm-prefill-step-size (loops at single-chunk prefill too,
  though degradation is milder)
- temperature (loops at 0.0, 0.7, 1.0)
- the ``vllm_mlx.patches.gemma4_mllm`` defensive-offset patch
  (loops with the patch and with the patch disabled)
- reasoning_content field on prior assistant turn

Root divergence: SimpleEngine routes text-only requests through
``mlx_lm.models.gemma4_text`` (via ``vllm_mlx.text_model_from_vlm``).
BatchedEngine runs the same request through ``mlx_vlm.models.gemma4.language``.
The two implementations differ in MoE routing (softmax-over-all-experts +
renormalize vs. softmax-over-top-k logits) and attention call ordering.

The test is an integration test — it requires a Gemma 4 server reachable at
``GEMMA4_BATCHED_URL`` (default ``http://127.0.0.1:8082``) launched with
``--continuous-batching``. It is skipped if no such server responds.
"""

import json
import os
import urllib.error
import urllib.request

import pytest

DEFAULT_URL = os.environ.get("GEMMA4_BATCHED_URL", "http://127.0.0.1:8082")
DEFAULT_MODEL = os.environ.get("GEMMA4_MODEL", "gemma26a4")

TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

# 106-line LS /tmp snapshot from the original repro (2026-05-22).
# Deterministic — inlined so the test is reproducible across machines.
_TOOL_RESULT = "\n".join(
    [
        f"__KMP_REGISTERED_LIB_{n}"
        for n in (
            15661, 17314, 18914, 19650, 20146, 22267, 24046, 25096, 25694, 27309,
            28161, 28964, 30688, 32421, 33122, 33236, 34473, 35382, 35420, 37138,
            37522, 37588, 42517, 45393, 46371, 46449, 47006, 47056, 47150, 47896,
            47958, 48399, 48470, 49605, 49651, 50398, 50814, 55074, 55151, 62315,
            65880, 68159, 68221, 68262, 68446, 68751, 68824, 68946, 69026, 69125,
            69158, 69207, 69340, 69379, 71620, 72898, 74587, 78298, 80438, 80771,
            83921, 86286, 8994,
        )
    ]
    + [
        "asitop_powermetrics1778964863",
        "bench_mtp_results.txt",
        "bench-baseline.log",
        "bench-mtp.log",
        "cache-watch.pid",
        "cache-watch.stdout.log",
        "chat-dense.out",
        "chat-dense2.out",
        "chat-launch.out",
        "chat-launch2.out",
        "chat-wrapup.out",
        "claude-501",
        "com.apple.launchd.vuGT5OT54C",
        "ticket-0RNPdM",
        "ticket-oiCm1L",
        "watchdog.out",
        "wrapup.out",
    ]
)


def _server_responsive(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


def _chat(url: str, model: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def _looks_like_token_loop(text: str) -> tuple[bool, str]:
    """Heuristic: detect degenerate token-repetition output.

    Returns (is_loop, reason). A healthy model answers in a sentence; a
    looping model emits the same 3-8 char substring many times in a row.
    """
    if not text:
        return False, "empty"
    # Same short n-gram repeated >=8x consecutively is degenerate.
    for n in (3, 4, 5, 6):
        for i in range(len(text) - n * 8):
            gram = text[i : i + n]
            if gram.strip() == "":
                continue
            if text[i : i + n * 8] == gram * 8:
                return True, f"{gram!r} repeated 8x at offset {i}"
    return False, "ok"


@pytest.fixture(scope="module")
def gemma_url():
    if not _server_responsive(DEFAULT_URL):
        pytest.skip(
            f"No Gemma 4 BatchedEngine server at {DEFAULT_URL}. "
            "Start one with: vllm-mlx serve <gemma-4-26b-a4b-it> "
            "--continuous-batching --tool-call-parser gemma4 "
            "--reasoning-parser gemma4 --port 8082"
        )
    return DEFAULT_URL


def test_tool_result_followup_does_not_loop(gemma_url):
    """The bug: passing back a long tool result triggers a token loop.

    Replays the canonical 3-turn flow:
        user → assistant(tool_call) → tool(result) → [server samples next turn]

    Asserts:
        - server returns 200
        - response does not contain a degenerate token loop in either
          ``content`` or ``reasoning_content``
        - finish_reason is not ``length`` for a generous max_tokens budget
          (model under load should naturally stop after a one-sentence answer)
    """
    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": 256,
        "temperature": 0.0,
        "tools": [TOOL],
        "messages": [
            {
                "role": "user",
                "content": (
                    "Are any files relating to asitop in /tmp? Answer in one sentence."
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": '{"path":"/tmp"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": _TOOL_RESULT,
            },
        ],
    }

    r = _chat(gemma_url, DEFAULT_MODEL, payload)
    choice = r["choices"][0]
    msg = choice["message"]
    fr = choice["finish_reason"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    content_loop, content_why = _looks_like_token_loop(content)
    reason_loop, reason_why = _looks_like_token_loop(reasoning)

    assert not content_loop, (
        f"content contains a degenerate token loop ({content_why}). "
        f"finish_reason={fr}. content[:300]={content[:300]!r}"
    )
    assert not reason_loop, (
        f"reasoning_content contains a degenerate token loop ({reason_why}). "
        f"finish_reason={fr}. reasoning[:300]={reasoning[:300]!r}"
    )
    assert fr != "length", (
        "Model hit max_tokens for a one-sentence answer — strong signal of "
        f"a generation loop. content[:300]={content[:300]!r} "
        f"reasoning[:300]={reasoning[:300]!r}"
    )


def test_tool_result_followup_does_not_loop_past_sliding_window(gemma_url):
    """The harder case: tool result that pushes prompt past Gemma 4's
    sliding-window cap (1024 tokens) and the sliding RotatingKVCache must
    rotate during prefill.

    A prior version of ``_trim_rotating_caches`` clamped
    ``RotatingKVCache.offset`` to ``max_size`` before merging caches into
    ``BatchRotatingKVCache``. ``offset`` is the absolute token-position
    counter (not a buffer index), so clamping it silently rewound the RoPE
    position seen by the next generated token from ~1700 back to 1024 —
    queries and cached keys then sat in different rotary coordinate
    systems and decode produced sentence-level loops drawn from prompt
    tokens. This test guards against that regression.
    """
    long_files = [f"file_{i:04d}.txt" for i in range(250)] + ["asitop_test.log"]
    long_result = "\n".join(long_files)

    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": 256,
        "temperature": 0.0,
        "tools": [TOOL],
        "messages": [
            {
                "role": "user",
                "content": (
                    "Are any files relating to asitop in /tmp? Answer in one sentence."
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_long",
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "arguments": '{"path":"/tmp"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_long",
                "content": long_result,
            },
        ],
    }

    r = _chat(gemma_url, DEFAULT_MODEL, payload)
    choice = r["choices"][0]
    msg = choice["message"]
    fr = choice["finish_reason"]
    ptok = r.get("usage", {}).get("prompt_tokens", 0)
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""

    # Sanity: prompt must be longer than Gemma 4's sliding window (1024) for
    # this test to actually exercise the cache-rotation path.
    assert ptok > 1024, (
        f"Test prompt is only {ptok} tokens — needs to exceed Gemma 4's "
        "sliding window (1024) to hit the cache-rotation path being tested."
    )

    content_loop, content_why = _looks_like_token_loop(content)
    reason_loop, reason_why = _looks_like_token_loop(reasoning)

    assert not content_loop, (
        f"content loop at ptok={ptok} ({content_why}). "
        f"finish={fr}. content[:300]={content[:300]!r}"
    )
    assert not reason_loop, (
        f"reasoning loop at ptok={ptok} ({reason_why}). "
        f"finish={fr}. reasoning[:300]={reasoning[:300]!r}"
    )
    assert fr != "length", (
        f"Hit max_tokens at ptok={ptok} for a one-sentence answer — loop "
        f"likely. content[:300]={content[:300]!r} reasoning[:300]={reasoning[:300]!r}"
    )
    # Coherence check: the model should reference the planted "asitop"
    # filename or at minimum produce a yes/no answer. Looping outputs
    # like ``file_15.txt file_15.txt …`` will fail this.
    haystack = (content + " " + reasoning).lower()
    assert "asitop" in haystack or "no" in haystack.split()[:5] or "yes" in haystack.split()[:5], (
        f"Output does not reference 'asitop' or contain a yes/no — possible "
        f"loop. content[:300]={content[:300]!r} reasoning[:300]={reasoning[:300]!r}"
    )
