#!/bin/bash
# nightly-check.sh - light regression check of the fork build that is serving on this box.
#
#   scripts/fork/nightly-check.sh               run, compare with the recorded baseline
#   scripts/fork/nightly-check.sh --rebaseline  run, and record the results as the new baseline
#
# Steps, all on ONE free GPU taken with gpu-lock (skipped when both are busy):
#   1. ctest -L main
#   2. the fork's own GPU tests again with each fusion / cache / graph switch turned off
#   3. greedy output (temp 0, 48 tokens) of a few models: md5 against the baseline
#   4. perplexity of one model on a fixed text: against the baseline, within PPL_TOL
#   5. long-context canary: a ~25k token prompt with tools must end in a read_file call on a doc,
#      not `////` (the symptom of 2026-09-30, see KV-ARCH-INVESTIGATION.md)
#
# A changed md5 is not necessarily a bug (an intentional numeric change also moves it): it is
# flagged for a person, who reruns with --rebaseline once satisfied. Everything else is a
# failure. The report goes to $NIGHTLY_DIR/<date>.md, a summary line to $NIGHTLY_DIR/history.tsv.
# Nothing here stresses a card beyond normal serving (see GPU-INCIDENTS.md: no burn-in tests).
#
# Exit codes: 0 all passed, 1 a check failed, 3 only md5 changes, 4 skipped (no free GPU).

set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="${BIN:-$REPO/build/bin}"
NIGHTLY_DIR="${NIGHTLY_DIR:-$HOME/src/llama.cpp/nightly}"
GPU_LOCK="${GPU_LOCK:-$HOME/src/ollama-turboquant/gpu-lock}"
GGUF="${GGUF:-/home/adderek/src/vLLM/models/gguf}"
PPL_TOL="${PPL_TOL:-0.02}"
# name|path|extra args for the md5 check
MODELS=(
  "ornith35|$GGUF/ornith-1.0-35b-Q4_K_M.gguf|-ctk turbo4 -ctv turbo4"
  "qwen38-27b|$GGUF/Qwen3.8-27B-IQ4_NL.gguf|-ctk turbo4 -ctv turbo4"
  "gemma4-26b|$GGUF/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf|"
  "ministral8b|$GGUF/ministral-3-8b-instruct/mistralai_Ministral-3-8B-Instruct-2512-IQ4_NL.gguf|-ctk turbo4 -ctv turbo4"
)
PPL_MODEL="$GGUF/ornith-1.0-35b-Q4_K_M.gguf"
CANARY_MODEL="$GGUF/Qwen3.8-27B-IQ4_NL.gguf"
CANARY_PORT="${CANARY_PORT:-18097}"

rebaseline=0
[ "${1:-}" = "--rebaseline" ] && rebaseline=1

mkdir -p "$NIGHTLY_DIR"
STAMP="$(date +%F_%H%M)"
REPORT="$NIGHTLY_DIR/$STAMP.md"
BASE="$NIGHTLY_DIR/baseline.tsv"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"; [ -n "${GPU:-}" ] && "$GPU_LOCK" release "$GPU" --pid $$ >/dev/null 2>&1' EXIT

n_fail=0; n_changed=0
say()   { echo "$*" | tee -a "$REPORT" >&2; }
ok()    { say "- OK      $*"; }
fail()  { say "- FAIL    $*"; n_fail=$((n_fail + 1)); }
changed() { say "- CHANGED $*"; n_changed=$((n_changed + 1)); }
base_get() { [ -f "$BASE" ] && awk -F'\t' -v k="$1" '$1==k {print $2}' "$BASE" | tail -1; }
base_set() { NEW_BASE+=("$1"$'\t'"$2"); }
NEW_BASE=()

commit="$(git -C "$REPO" log --oneline -1 2>/dev/null)"
say "# nightly-check $STAMP"
say ""
say "build: \`$BIN\`, commit \`$commit\`"

# --- take a free GPU ----------------------------------------------------------
GPU=""
for g in gpu0 gpu1; do
  if "$GPU_LOCK" claim "$g" --pid $$ --who "nightly-check" --note "regression check, ~20 min" 2>/dev/null; then GPU="$g"; break; fi
done
if [ -z "$GPU" ]; then
  say "skipped: no free GPU"; "$GPU_LOCK" status >> "$REPORT"
  printf "%s\t%s\t%s\n" "$STAMP" "skipped" "$commit" >> "$NIGHTLY_DIR/history.tsv"
  exit 4
fi
export HIP_VISIBLE_DEVICES="${GPU#gpu}"
# ROCm index follows PCI bus order; unique_id names the physical card (GPU-INCIDENTS.md)
uid="$(for f in /sys/class/drm/card*/device/unique_id; do echo "$(cat "$f") $(basename "$(readlink -f "${f%/unique_id}")")"; done | sort -k2 | sed -n "$((${GPU#gpu} + 1))p" | cut -d' ' -f1)"
say "gpu: $GPU (unique_id $uid), uptime: $(uptime -p)"
say ""

# --- 1. ctest -------------------------------------------------------------------
say "## ctest"
if (cd "$REPO/build" && ctest -L main -E ggml-vocabs -j4 > "$WORK/ctest.log" 2>&1); then
  ok "ctest -L main: $(grep -oE '[0-9]+% tests passed.*' "$WORK/ctest.log")"
else
  fail "ctest -L main: $(grep -oE '[0-9]+% tests passed.*' "$WORK/ctest.log")"
  grep -E "\*\*\*|Failed" "$WORK/ctest.log" | head -10 | sed 's/^/      /' | tee -a "$REPORT" >&2
fi

# --- 2. fork GPU tests with each switch off -------------------------------------
say ""; say "## fork tests, switches off"
FORK_TESTS="mmvq-src1|sigmoid-gate|add-unary|gdn-beta|fattn|turbo|top-k-graph"
for env in GGML_CUDA_DISABLE_FUSION=1 GGML_CUDA_MMVQ_SRC1_CACHE=0 GGML_CUDA_DISABLE_GRAPHS=1; do
  if (cd "$REPO/build" && env "$env" ctest -R "$FORK_TESTS" > "$WORK/fork.log" 2>&1); then
    ok "$env: $(grep -oE '[0-9]+% tests passed.*' "$WORK/fork.log")"
  else
    fail "$env: $(grep -oE '[0-9]+% tests passed.*' "$WORK/fork.log")"
  fi
done

# --- 3. greedy output md5 -------------------------------------------------------
say ""; say "## greedy output"
for e in "${MODELS[@]}"; do
  IFS='|' read -r name path args <<< "$e"
  [ -f "$path" ] || { say "- skip    $name: $path missing"; continue; }
  # shellcheck disable=SC2086
  md5="$("$BIN/llama-completion" -m "$path" -ngl 99 -fa 1 $args --temp 0 -n 48 -no-cnv \
          -p "Explain in detail how a hash map handles collisions:" 2>/dev/null | md5sum | cut -c1-32)"
  if [ "$md5" = "d41d8cd98f00b204e9800998ecf8427e" ]; then fail "$name: empty output"; continue; fi
  base_set "md5.$name" "$md5"
  old="$(base_get "md5.$name")"
  if [ -z "$old" ]; then ok "$name: $md5 (no baseline yet)"
  elif [ "$old" = "$md5" ]; then ok "$name: $md5"
  else changed "$name: $md5, baseline $old"; fi
done

# --- 4. perplexity --------------------------------------------------------------
say ""; say "## perplexity"
cat "$REPO"/docs/build.md "$REPO"/docs/function-calling.md "$REPO"/README.md > "$WORK/ppl.txt" 2>/dev/null
ppl="$("$BIN/llama-perplexity" -m "$PPL_MODEL" -ngl 99 -fa 1 -ctk turbo4 -ctv turbo4 -c 512 --chunks 6 \
        -f "$WORK/ppl.txt" 2>&1 | grep -oE 'Final estimate: PPL = [0-9.]+' | grep -oE '[0-9.]+$')"
if [ -z "$ppl" ]; then fail "perplexity did not run"
else
  base_set "ppl.ornith35" "$ppl"
  old="$(base_get "ppl.ornith35")"
  if [ -z "$old" ]; then ok "ornith35 PPL $ppl (no baseline yet)"
  elif awk -v a="$ppl" -v b="$old" -v t="$PPL_TOL" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<=t)}'; then ok "ornith35 PPL $ppl (baseline $old)"
  else fail "ornith35 PPL $ppl, baseline $old (tolerance $PPL_TOL)"; fi
fi

# --- 5. long-context canary -----------------------------------------------------
say ""; say "## long-context canary"
python3 - "$REPO" "$WORK/canary.json" <<'EOF'
import sys, json, glob
repo, out = sys.argv[1], sys.argv[2]
doc = "".join(open(f).read() for f in sorted(glob.glob(repo + "/docs/*.md")))[:90000]
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
json.dump({"messages": [{"role": "system", "content": "You are a coding agent. Use tools when needed."},
                        {"role": "user", "content": "Docs:\n" + doc + "\n\nWhich file would you read to learn how to build with ROCm? Call the tool."}],
           "tools": tools, "max_tokens": 400, "temperature": 0}, open(out, "w"))
EOF
"$BIN/llama-server" -m "$CANARY_MODEL" --port "$CANARY_PORT" --host 127.0.0.1 -c 32768 -ngl 99 -fa on \
  -ctk turbo4 -ctv turbo4 --jinja -np 1 -b 4096 -ub 4096 --reasoning-budget 256 --offline > "$WORK/canary-server.log" 2>&1 &
spid=$!
for _ in $(seq 1 180); do curl -sf "localhost:$CANARY_PORT/health" >/dev/null && break; kill -0 $spid 2>/dev/null || break; sleep 1; done
resp="$(curl -s --max-time 600 "localhost:$CANARY_PORT/v1/chat/completions" -H 'Content-Type: application/json' -d @"$WORK/canary.json")"
kill $spid 2>/dev/null; wait $spid 2>/dev/null
verdict="$(echo "$resp" | python3 -c '
import sys, json
try:
    j = json.load(sys.stdin); m = j["choices"][0]["message"]
except Exception as e:
    print("FAIL no response"); sys.exit()
text = (m.get("reasoning_content") or "") + (m.get("content") or "")
calls = [c["function"]["name"] + " " + c["function"]["arguments"] for c in (m.get("tool_calls") or [])]
n = j.get("usage", {}).get("prompt_tokens")
if "////" in text: print(f"FAIL {n} tokens: output is ////")
elif any(c.startswith("read_file") and ".md" in c for c in calls): print(f"OK {n} tokens: {calls[0][:80]}")
else: print(f"FAIL {n} tokens: no read_file call on a .md path: {calls} {text[:80]!r}")
')"
case "$verdict" in OK*) ok "canary ${verdict#OK }" ;; *) fail "canary ${verdict#FAIL } (if the other GPU passes this, record it in GPU-INCIDENTS.md)" ;; esac
grep -m1 -E "Memory access fault|HW Exception" "$WORK/canary-server.log" | sed 's/^/      /' | tee -a "$REPORT" >&2

# --- summary ---------------------------------------------------------------------
if [ "$rebaseline" = 1 ] || [ ! -f "$BASE" ]; then
  printf "%s\n" "${NEW_BASE[@]}" > "$BASE"
  say ""; say "baseline written: $BASE"
fi
status=pass; code=0
[ "$n_changed" -gt 0 ] && { status=changed; code=3; }
[ "$n_fail" -gt 0 ] && { status=fail; code=1; }
say ""; say "result: $status ($n_fail failed, $n_changed changed)"
printf "%s\t%s\t%s\t%s\t%s\n" "$STAMP" "$status" "$GPU" "$uid" "$commit" >> "$NIGHTLY_DIR/history.tsv"
exit $code
