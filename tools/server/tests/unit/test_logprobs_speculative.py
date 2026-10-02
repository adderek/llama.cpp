#!/usr/bin/env python
# Fork feature: tokens accepted from a speculative draft carry the same probs as tokens
# generated one by one (read from the target model's verification logits).
import pytest

# ensure grandparent path is in sys.path
from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import *

server: ServerProcess

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.mark.parametrize("post_sampling_probs", [False, True])
def test_logprobs_speculative_match_plain(post_sampling_probs: bool):
    global server
    server.start()
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "n_predict": 48,
        "n_probs": 3,
        "post_sampling_probs": post_sampling_probs,
        "cache_prompt": False,
    }
    res_plain = server.make_request("POST", "/completion", data=request | {"speculative.type": "none"})
    res_spec  = server.make_request("POST", "/completion", data=request)
    assert res_plain.status_code == 200 and res_spec.status_code == 200
    assert res_plain.body["timings"].get("draft_n", 0) == 0  # "none" turned drafting off
    assert res_spec.body["timings"]["draft_n_accepted"] > 0

    key, top = ("prob", "top_probs") if post_sampling_probs else ("logprob", "top_logprobs")
    plain = res_plain.body["completion_probabilities"]
    spec  = res_spec.body["completion_probabilities"]
    assert [t["id"] for t in spec] == [t["id"] for t in plain]
    for a, b in zip(plain, spec):
        assert len(b[top]) > 0
        assert [t["id"] for t in b[top]] == [t["id"] for t in a[top]]
        # verifying a batch of draft tokens and decoding them one by one take different matmul
        # paths; on this tiny quantized model logprobs differ by up to ~3e-3 (measured)
        assert b[key] == pytest.approx(a[key], abs=1e-2)


def test_speculative_type_unknown_is_rejected():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 4,
        "speculative.type": "bogus",
    })
    assert res.status_code == 400


def test_speculative_type_not_enabled_is_rejected():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_predict": 4,
        "speculative.type": "ngram-cache",
    })
    assert res.status_code == 400
