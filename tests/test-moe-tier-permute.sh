#!/usr/bin/env bash
# Used by test-moe-tier through TEST_MOE_TIER_PERMUTE: permute MODEL into OUT with tools/moe-tier,
# from a made-up imatrix whose counts reverse the expert order of every layer.
set -euo pipefail
model=$1 out=$2
root=$(cd "$(dirname "$0")/.." && pwd)
imatrix=$(mktemp --suffix=.gguf)
trap 'rm -f "$imatrix"' EXIT

uv run --quiet --with numpy --with pyyaml --with tqdm python3 - "$model" "$imatrix" <<PY
import sys
sys.path.insert(0, "$root/gguf-py")
import numpy as np, gguf
r = gguf.GGUFReader(sys.argv[1])
arch = r.fields["general.architecture"].contents()
n_expert = int(r.fields[f"{arch}.expert_count"].contents())
w = gguf.GGUFWriter(sys.argv[2], arch="imatrix")
for t in r.tensors:
    if t.name.endswith("_exps.weight"):
        # a different order per layer, none of them the identity
        il = int(t.name.split(".")[1])
        counts = np.roll(np.arange(n_expert, dtype=np.float32), il + 3)
        w.add_tensor(t.name + ".counts", counts.reshape(n_expert, 1))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
PY

rm -f "$out"
uv run --quiet --script "$root/tools/moe-tier/moe-tier.py" permute "$model" "$imatrix" "$out" >/dev/null
