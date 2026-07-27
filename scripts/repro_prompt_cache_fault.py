#!/usr/bin/env python3
"""Drive llama-server so that it repeatedly saves and restores full slot states
through the prompt cache - the path that faulted the GPU (SDMA read of a freed
host buffer).

More distinct conversations than slots => every round the LRU picks a slot whose
previous occupant has to be saved, and the returning conversation is restored
from the cache.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog while the maintainer reviews "
    "another patch in the queue and the build server hums along in the corner. "
)


def prefix(tag: str, words: int) -> str:
    # unique per conversation, so no two share an LCP worth reusing
    return f"conversation {tag}: " + (FILLER * words)


def post(url: str, payload: dict, timeout: int):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--convs", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--filler", type=int, default=250, help="filler repeats (~14 tok each)")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    tags = [chr(ord("A") + i) for i in range(args.convs)]
    history = {t: [{"role": "user", "content": prefix(t, args.filler)}] for t in tags}

    for rnd in range(args.rounds):
        for t in tags:
            msgs = history[t] + [{"role": "user", "content": f"round {rnd}: reply with one short sentence."}]
            t0 = time.time()
            try:
                res = post(
                    f"{args.url}/v1/chat/completions",
                    {"messages": msgs, "max_tokens": 16, "temperature": 0.1, "cache_prompt": True},
                    args.timeout,
                )
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                print(f"round {rnd} conv {t}: REQUEST FAILED after {time.time()-t0:.1f}s: {e}", flush=True)
                return 1

            txt = res["choices"][0]["message"]["content"]
            history[t] = msgs + [{"role": "assistant", "content": txt}]
            print(f"round {rnd} conv {t}: ok in {time.time()-t0:6.1f}s  ({txt[:40]!r})", flush=True)

    print("done, no failures", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
