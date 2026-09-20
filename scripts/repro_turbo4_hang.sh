#!/usr/bin/env bash
# Repro attempt for BUG_GPU_HANG_turbo4.md: drive a long context (~29k
# tokens) through several context-checkpoint creations with turbo4 KV cache,
# then issue a follow-up decode and watch for a hang (zero tokens, GPU idle,
# host blocked in hsaKmtWaitOnEvent).
#
# Usage: ./repro_turbo4_hang.sh <model.gguf> <port>
# While it's running (or hung), in another shell:
#   rocm-smi --showuse                     # GPU%  should go to 0 if hung
#   gdb -p <server_pid> -batch -ex "thread apply all bt" | grep -A3 "libhsa-runtime"
#   sudo dmesg | tail -50 | grep -iE 'amdgpu|fault|reset|kfd'   # needs sudo
set -euo pipefail

MODEL="${1:?usage: $0 <model.gguf> <port>}"
PORT="${2:?usage: $0 <model.gguf> <port>}"
BIN="$(dirname "$0")/../build/bin/llama-server"

LOG=/tmp/repro_turbo4_hang.log
"$BIN" -m "$MODEL" --port "$PORT" \
  --cache-type-k turbo4 --cache-type-v turbo4 \
  -c 32768 -fa on -ngl 99 -np 1 -b 4096 -ub 4096 -t 12 \
  --cont-batching --jinja --ctx-checkpoints 4 --ctx-checkpoint-min-step 4096 \
  > "$LOG" 2>&1 &
SRV_PID=$!
echo "server pid: $SRV_PID, log: $LOG"

for i in $(seq 1 30); do
  sleep 2
  curl -s -m 2 "http://localhost:$PORT/health" 2>/dev/null | grep -q ok && break
done

echo "server ready, building up context in ~4k-token increments..."

# Build a long, repetitive-but-unique prompt in chunks so each request grows
# total context by roughly one checkpoint-min-step, forcing several
# create_checkpoint() calls before the final large decode.
PROMPT_CHUNK=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * 90))")

CONV_FILE=/tmp/repro_turbo4_messages.json
python3 - "$CONV_FILE" <<'PYEOF'
import json, sys
json.dump([], open(sys.argv[1], "w"))
PYEOF

for round in $(seq 1 8); do
  python3 - "$CONV_FILE" "$PROMPT_CHUNK" "$round" <<'PYEOF'
import json, sys
path, chunk, rnd = sys.argv[1], sys.argv[2], sys.argv[3]
msgs = json.load(open(path))
msgs.append({"role": "user", "content": f"[round {rnd}] Repeat back this text verbatim, then continue the story by one sentence: {chunk}"})
json.dump(msgs, open(path, "w"))
PYEOF
  echo "--- round $round: sending request, current total context growing ---"
  RESP=$(curl -s -m 120 "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"repro\",\"messages\":$(cat "$CONV_FILE"),\"max_tokens\":150,\"stream\":false}")
  CONTENT=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "REQUEST_FAILED_OR_HUNG")
  if [ "$CONTENT" = "REQUEST_FAILED_OR_HUNG" ]; then
    echo "!!! Request $round did not return within 120s - possible hang. Check:"
    echo "  tail -20 $LOG"
    echo "  rocm-smi --showuse"
    echo "  gdb -p $SRV_PID -batch -ex 'thread apply all bt'"
    exit 1
  fi
  python3 - "$CONV_FILE" "$CONTENT" <<'PYEOF'
import json, sys
path, content = sys.argv[1], sys.argv[2]
msgs = json.load(open(path))
msgs.append({"role": "assistant", "content": content})
json.dump(msgs, open(path, "w"))
PYEOF
  grep -c "created context checkpoint" "$LOG" | xargs echo "checkpoints created so far:"
done

echo "All 8 rounds completed without a hang. Repro did not trigger this run."
echo "Server still running (pid $SRV_PID) for manual follow-up. Kill with: kill $SRV_PID"
