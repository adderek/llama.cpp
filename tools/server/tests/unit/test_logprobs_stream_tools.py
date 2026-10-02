#!/usr/bin/env python
# Fork feature: logprobs in streamed chat completions with tools.
# Every generated token must appear exactly once in the stream's logprobs, including the tokens
# whose text the tool-call / reasoning parser holds back and emits later (or never, as markup).
import pytest

# ensure grandparent path is in sys.path
from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import *

server: ServerProcess

TOOL = {
    "type": "function",
    "function": {
        "name": "test",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {"success": {"type": "boolean", "const": True}},
            "required": ["success"],
        },
    },
}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_ctx = 8192
    server.n_batch = 2048
    server.jinja = True
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'


# the tiny model does not call tools by itself; an assistant prefill that opens a tool call
# makes the parser hold back most of the generated tokens, which is the case under test
PREFILL_TOOL_CALL = '<tool_call>\n{"name": "test", "arguments": {"success": '


def request_body(n_predict: int, prefill: str | None) -> dict:
    messages = [{"role": "user", "content": "Call the test tool"}]
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})
    return {
        "max_tokens": n_predict,
        "messages": messages,
        "tools": [TOOL],
        "tool_choice": "required",
        "temperature": 0.0,
        "top_k": 1,
        "logprobs": True,
        "top_logprobs": 5,
        "cache_prompt": False,
    }


@pytest.mark.parametrize("n_predict,prefill", [
    (64,  None),
    (32,  PREFILL_TOOL_CALL),
    (128, PREFILL_TOOL_CALL),
])
def test_logprobs_stream_tools_every_token_once(n_predict: int, prefill: str | None):
    global server
    server.start()

    res = server.make_request("POST", "/chat/completions", data=request_body(n_predict, prefill))
    assert res.status_code == 200, res.body
    ns_choice = res.body["choices"][0]
    ns_ids = [t["id"] for t in ns_choice["logprobs"]["content"]]
    assert len(ns_ids) == res.body["usage"]["completion_tokens"]

    data = request_body(n_predict, prefill) | {"stream": True, "stream_options": {"include_usage": True}}
    ids = []
    n_tool_call_rows = 0
    usage = None
    for chunk in server.make_stream_request("POST", "/chat/completions", data=data):
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk["choices"]:
            content = (choice.get("logprobs") or {}).get("content") or []
            for token in content:
                assert len(token["top_logprobs"]) == 5
                ids.append(token["id"])
            if choice["delta"].get("tool_calls"):
                n_tool_call_rows += len(content)

    assert usage is not None
    assert len(ids) == usage["completion_tokens"]
    assert ids == ns_ids  # same tokens, no loss, no duplicates
    if prefill is not None:
        assert n_tool_call_rows > 0


@pytest.mark.parametrize("stream", [False, True])
def test_logprobs_rows_describe_their_own_token(stream: bool):
    # each row's "token"/"bytes" is the token's own text, not the text emitted at its step:
    # a stop word holds back the tokens that form its prefix, which must not shift text
    # between rows.
    global server
    server.start()
    base = {
        "max_tokens": 48,
        "messages": [{"role": "user", "content": "Count from 1 to 30, separated by spaces."}],
        "temperature": 0.0,
        "top_k": 1,
        "logprobs": True,
        "top_logprobs": 1,
        "cache_prompt": False,
    }
    res = server.make_request("POST", "/chat/completions", data=base)
    assert res.status_code == 200, res.body
    text = res.body["choices"][0]["message"]["content"]
    assert len(text) > 24
    stop = text[16:20]  # spans a token boundary in practice, so a prefix is held back

    data = base | {"stop": [stop]}
    if stream:
        rows, content = [], ""
        for chunk in server.make_stream_request("POST", "/chat/completions", data=data | {"stream": True}):
            for choice in chunk["choices"]:
                content += choice["delta"].get("content") or ""
                rows += (choice.get("logprobs") or {}).get("content") or []
    else:
        res = server.make_request("POST", "/chat/completions", data=data)
        assert res.status_code == 200, res.body
        content = res.body["choices"][0]["message"]["content"]
        rows = res.body["choices"][0]["logprobs"]["content"]

    assert stop not in content
    assert len(rows) > 0
    for r in rows:
        piece = server.make_request("POST", "/detokenize", data={"tokens": [r["id"]]}).body["content"]
        assert bytes(r["bytes"]) == piece.encode(), (r, piece)


def test_logprobs_fast_path_matches_full_sort():
    # top_logprobs <= 32 takes the two-pass path over the logits; more falls back to sorting a copy
    # of the whole vocabulary. Both must give the same tokens and log-probabilities.
    global server
    server.start()
    def run(n):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is", "n_predict": 24, "temperature": 0.0,
            "n_probs": n, "cache_prompt": False,
        })
        assert res.status_code == 200, res.body
        return res.body["completion_probabilities"]
    fast, full = run(5), run(40)
    assert [t["id"] for t in fast] == [t["id"] for t in full]
    for a, b in zip(fast, full):
        assert a["logprob"] == pytest.approx(b["logprob"], abs=1e-4)
        assert len(a["top_logprobs"]) == 5
        for x, y in zip(a["top_logprobs"], b["top_logprobs"][:5]):
            assert x["id"] == y["id"]
            assert x["logprob"] == pytest.approx(y["logprob"], abs=1e-4)
