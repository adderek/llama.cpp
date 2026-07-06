#!/usr/bin/env python3
"""Deterministic repro for BUG_GPU_HANG_turbo4.md - v3, MULTI-TURN restore.

Root understanding (server-context.cpp:3643-3647): context checkpoints form
ONLY at user-message boundaries (is_user_start) or near prompt end - never
sprinkled through one big single-turn prompt. So to get checkpoints at multiple
positions (needed so a divergence has a checkpoint BEFORE it -> load_tgt restore
instead of do_reset full-reprocess), use a MULTI-TURN conversation.

v3 design:
  - FIXED prefix: N turns of (large user msg + short assistant reply), identical
    every round -> cached, checkpoints created at each turn boundary.
  - VARYING final turn: a large, unique user message each round -> diverges at
    the last turn boundary, server restores the checkpoint just before it
    (load_tgt) and decodes the big new turn. That RESTORE + large turbo4 decode
    is the exact hang trigger.

Watch for a request that never returns while GPU% -> 0.
Usage: repro_turbo4_hang.py --port 8093 [--rounds 40]
"""
import argparse, json, time, sys, urllib.request, urllib.error

def gen(seed, n):
    return " ".join(f"{seed}-{i}:{(i*2654435761)&0xffffff}" for i in range(n))

def chat(port, messages, max_tokens, timeout):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type":"application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return time.monotonic()-t0, d.get("usage",{}).get("completion_tokens",0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--fixed-turns", type=int, default=6)     # each -> a checkpoint boundary
    ap.add_argument("--turn-items", type=int, default=900)    # ~size of each fixed user turn
    ap.add_argument("--final-items", type=int, default=1800)  # LARGE divergent final turn -> big decode
    ap.add_argument("--timeout", type=int, default=200)
    ap.add_argument("--tag", default="", help="unique prefix so concurrent clients use independent slots")
    args = ap.parse_args()
    T = args.tag

    # Fixed multi-turn prefix (identical every round -> cached, builds checkpoints
    # at each user-message boundary). --tag makes each concurrent client distinct.
    fixed = []
    for t in range(args.fixed_turns):
        fixed.append({"role":"user","content":f"[{T}] Block {t}. Data: {gen(T+'u'+str(t), args.turn_items)}. Acknowledge with the block number."})
        fixed.append({"role":"assistant","content":f"Acknowledged block {t}."})

    print(f"[repro v3] {args.fixed_turns} fixed turns (~{args.turn_items} items each) "
          f"+ large divergent final turn (~{args.final_items} items); {args.rounds} rounds.", flush=True)

    for rnd in range(args.rounds):
        branch = ["beta","gamma","delta","epsilon"][rnd % 4]
        final = {"role":"user","content":
                 f"[{T}] Final divergent block [{branch} #{rnd}]: {gen(T+branch, args.final_items)}. "
                 f"Reply with just the first token of this block."}
        messages = fixed + [final]
        try:
            dt, ct = chat(args.port, messages, max_tokens=48, timeout=args.timeout)
            print(f"[repro] round {rnd:3d} {branch:8s} ok {dt:6.1f}s {ct} tok", flush=True)
            continue
        except urllib.error.HTTPError as he:
            # a fast HTTP error (e.g. 400 context-overflow) is a CONFIG problem, NOT a hang
            body = ""
            try: body = he.read().decode()[:200]
            except Exception: pass
            print(f"[repro] round {rnd:3d} {branch:8s} HTTP {he.code} (config err, not a hang): {body}", flush=True)
            sys.exit(2)
        except Exception as e:
            # timeout / connection failure with no fast response = the lost-wakeup HANG
            print(f"\n!!! round {rnd} DID NOT RETURN within {args.timeout}s: {type(e).__name__}: {e}", flush=True)
            print("!!! LIKELY THE HANG - inspect the wedged server (do not kill yet):", flush=True)
            print(f"    tail -30 ~/src/llama-serve/logs/qwen36-turbo4-repro.log", flush=True)
            print(f"    rocm-smi --showuse", flush=True)
            print(f"    SRV=$(pgrep -f 'port {args.port}'); gdb -p $SRV -batch -ex 'thread apply all bt' 2>&1 | grep -A4 -iE 'libhsa|kfd_wait|hsaKmt'", flush=True)
            sys.exit(1)
    print("[repro] all rounds completed WITHOUT a hang.", flush=True)

if __name__ == "__main__":
    main()
