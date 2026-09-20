#!/usr/bin/env bash
# Deterministic repro driver for the turbo4 / spec / ctx-checkpoint stall.
# Drives a running llama-server with a long context that forces a context
# checkpoint rollover + restore, then a follow-up decode (handoff 4a).
#
# Detects a STALL: time-to-first-token or inter-token gap exceeding STALL_SECS.
# Does NOT launch the server (start it with the wedged launch command first,
# optionally with AMD_LOG_LEVEL=3 to capture the HSA trace).
#
# Usage:
#   ./repro_turbo4_hang.sh                 # default localhost:8081, 50 rounds
#   HOST=localhost PORT=8081 ROUNDS=200 STALL_SECS=30 ./repro_turbo4_hang.sh

set -u

HOST="${HOST:-localhost}"
PORT="${PORT:-8081}"
ROUNDS="${ROUNDS:-50}"
STALL_SECS="${STALL_SECS:-30}"
# Approx prompt sizes in tokens (filler is ~1 token/word).
BASE_TOKENS="${BASE_TOKENS:-17000}"
GROW_TOKENS="${GROW_TOKENS:-12000}"   # base+grow ~= 29k, crosses checkpoint 4
GEN_TOKENS="${GEN_TOKENS:-256}"

URL="http://${HOST}:${PORT}/v1/chat/completions"

if ! curl -s -m 3 "http://${HOST}:${PORT}/health" >/dev/null; then
    echo "FATAL: server not reachable at ${HOST}:${PORT}. Start it first." >&2
    exit 1
fi

CONCURRENCY="${CONCURRENCY:-3}"   # match -np; saturates slots / cont-batching

# Repetitive filler: short cycling vocab so the ngram-cache speculator actually
# drafts and verifies (the real spec path). distinct-word filler never drafts.
filler() {
    local n="$1"
    python3 - "$n" <<'PY'
import sys
n = int(sys.argv[1])
# 64-word cycling vocab with periodic phrases -> ngram-draftable, but offset
# so the long prefix still differs per round when a salt is appended.
vocab = ["the","model","reads","tokens","and","then","writes","cache",
         "block","after","block","while","the","server","keeps","the",
         "slot","busy","with","work","across","many","steps","of",
         "decode","until","the","context","grows","large","enough","to",
         "trigger","a","checkpoint","restore","followed","by","another",
         "long","decode","that","may","wedge","the","gpu","or","stall",
         "the","slot","without","ever","releasing","it","back","to",
         "the","queue","for","the","next","pending","request","now"]
sys.stdout.write(" ".join(vocab[i % len(vocab)] for i in range(n)))
PY
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# Round A: base context. Round B: shared prefix + growth (forces checkpoint restore).
{ filler "$BASE_TOKENS"; printf '  Summarize the above in one sentence.'; } > "$TMP/A"
{ filler "$BASE_TOKENS"; printf ' '; filler "$GROW_TOKENS"; printf '  Now summarize everything above.'; } > "$TMP/B"

# Stream a chat completion; print PASS/STALL based on token timing.
# Returns 2 on stall.
run_one() {
    local promptfile="$1" label="$2"
    local body
    # --rawfile avoids ARG_MAX: prompt read from file, not argv.
    body="$(jq -n --rawfile p "$promptfile" --argjson n "$GEN_TOKENS" \
        '{model:"x",stream:true,max_tokens:$n,temperature:0.1,
          messages:[{role:"user",content:$p}]}')"

    # curl streams SSE; @file body also avoids ARG_MAX on the request.
    printf '%s' "$body" | curl -sN -m 600 "$URL" -H 'Content-Type: application/json' --data-binary @- \
    | STALL_SECS="$STALL_SECS" python3 - "$label" <<'PY'
import sys, select, time, os
label = sys.argv[1]
stall = float(os.environ["STALL_SECS"])
last = time.time()
got = 0
while True:
    r,_,_ = select.select([sys.stdin], [], [], stall)
    if not r:
        print("STALL label=%s tokens=%d idle>%ss" % (label, got, stall), flush=True)
        sys.exit(2)
    line = sys.stdin.readline()
    if not line:
        print("DONE  label=%s tokens=%d" % (label, got), flush=True)
        sys.exit(0)
    if line.startswith("data:") and '"content"' in line:
        got += 1
        last = time.time()
PY
    return "${PIPESTATUS[1]}"
}

echo "repro start: ${ROUNDS} rounds x ${CONCURRENCY} clients, base~${BASE_TOKENS}t grow~${GROW_TOKENS}t, stall>${STALL_SECS}s"
for ((r=1; r<=ROUNDS; r++)); do
    echo "== round $r/$ROUNDS =="
    pids=()
    for ((c=1; c<=CONCURRENCY; c++)); do
        # Per-client salt at a varied position forces checkpoint restore to
        # diverge at different KV offsets each round -> exercises restore churn.
        f="$TMP/B_${r}_${c}"
        head -c $(( (BASE_TOKENS + r*131 + c*17) % (BASE_TOKENS*5) + 1 )) "$TMP/B" > "$f"
        printf ' salt-%d-%d  Now summarize everything above.' "$r" "$c" >> "$f"
        ( run_one "$f" "B${r}c${c}"; echo "$?" > "$f.rc" ) &
        pids+=($!)
    done
    wait "${pids[@]}"
    for ((c=1; c<=CONCURRENCY; c++)); do
        rc="$(cat "$TMP/B_${r}_${c}.rc" 2>/dev/null)"
        [ "$rc" = 2 ] && { echo "HANG on round $r client $c"; exit 2; }
    done
done
echo "no hang across ${ROUNDS} rounds x ${CONCURRENCY} clients"
