#!/bin/bash
# GPU correctness + speed check for LLAMA_MOE_HOT. Needs exclusive GPU.
# usage: gpu-check.sh PERMUTED_MODEL [OUT_DIR]
set -euo pipefail

M=${1:?permuted model}
OUT=${2:-/tmp/moe-tier-gpu-check}
BIN=$(cd "$(dirname "$0")/../../build-hip/bin" && pwd)
TEXT=${TEXT:-$HOME/src/ollama-turboquant/calib/calib-agentic.txt}
DEV=${DEV:-ROCm0}
HOTS=${HOTS:-"32 64 128 192"}

mkdir -p "$OUT"
head -c 60000 "$TEXT" > "$OUT/eval.txt"

ppl() { # name, extra args...
    local name=$1; shift
    "$BIN/llama-perplexity" -m "$M" -f "$OUT/eval.txt" -c 256 --chunks 4 -ngl 99 -dev "$DEV" "$@" > "$OUT/ppl-$name.log" 2>&1
    grep -E 'Mean +KLD|Same top p' "$OUT/ppl-$name.log" | sed "s/^/$name: /" >&2 || true
}

echo "== correctness (KLD vs all-GPU)" >&2
ppl gpu --kl-divergence-base "$OUT/base.bin"
ppl allcpu-exps -ot 'exps=CPU' --kl-divergence-base "$OUT/base.bin" --kl-divergence
for k in $HOTS; do
    LLAMA_MOE_HOT=$k ppl "hot$k" --kl-divergence-base "$OUT/base.bin" --kl-divergence
done

echo "== speed" >&2
bench() { # name, extra args...
    local name=$1; shift
    "$BIN/llama-bench" -m "$M" -ngl 99 -dev "$DEV" -p 512 -n 128 -r 3 -o csv "$@" 2> "$OUT/bench-$name.err" \
        | awk -F, -v n="$name" 'NR>1 {print n "," $0}'
}
{
    bench gpu
    bench allcpu-exps -ot 'exps=CPU'
    for k in $HOTS; do LLAMA_MOE_HOT=$k bench "hot$k"; done
} > "$OUT/bench.csv"
echo "wrote $OUT/bench.csv" >&2
