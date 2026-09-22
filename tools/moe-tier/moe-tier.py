#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyyaml", "tqdm"]
# ///
"""MoE expert tiering helpers.

  moe-tier.py stats   IMATRIX              per-expert routing counts as CSV on stdout,
                                           skew summary on stderr
  moe-tier.py permute MODEL IMATRIX OUT    rewrite MODEL with every MoE layer's experts
                                           reordered hottest-first

Experts are exchangeable: permuting the expert index of the *_exps tensors, the
router rows (ffn_gate_inp) and any per-expert bias by the same permutation leaves
the model's output unchanged. After `permute`, expert 0 of every layer is the one
the imatrix saw routed most often, so the hot set is a prefix of each fused expert
tensor and the cold set is a contiguous suffix - which is what lets the loader
place them on different backends (--moe-hot) and lets the cold suffix stay
mmap-backed.

The permutation is byte-level along the outermost (expert) dimension, so it works
on any quantisation type; quantise first, permute last.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def log(*a):
    print(*a, file=sys.stderr)


def read_counts(imatrix_path: str) -> dict[int, dict[str, np.ndarray]]:
    """layer -> {tensor role -> counts[n_expert]} for every *_exps tensor in the imatrix."""
    r = gguf.GGUFReader(imatrix_path)
    out: dict[int, dict[str, np.ndarray]] = {}
    for t in r.tensors:
        if not t.name.endswith(".weight.counts"):
            continue
        m = LAYER_RE.match(t.name[: -len(".weight.counts")])
        if not m or not m.group(2).endswith("_exps"):
            continue
        c = np.asarray(t.data, dtype=np.float64).reshape(-1)
        if c.size < 2:
            continue
        out.setdefault(int(m.group(1)), {})[m.group(2)] = c
    if not out:
        sys.exit(f"{imatrix_path}: no per-expert .counts tensors - not a MoE imatrix?")
    return out


def layer_order(roles: dict[str, np.ndarray]) -> np.ndarray:
    """Hottest-first permutation; ties keep the original index order."""
    total = sum(roles.values())
    return np.argsort(-total, kind="stable")


def cmd_stats(args) -> None:
    counts = read_counts(args.imatrix)
    print("layer,tensor,expert,count")
    for il in sorted(counts):
        for role in sorted(counts[il]):
            for e, c in enumerate(counts[il][role]):
                print(f"{il},{role},{e},{int(c)}")

    fracs = (0.10, 0.25, 0.50, 0.75)
    log("traffic share carried by the hottest X% of experts, per layer")
    log("layer  n_exp  " + "  ".join(f"top{int(f * 100):>3d}%" for f in fracs) + "   max/min   zero")
    for il in sorted(counts):
        total = sum(counts[il].values())
        s = np.sort(total)[::-1]
        cum = np.cumsum(s) / s.sum()
        n = s.size
        cells = "  ".join(f"{cum[max(1, int(round(f * n))) - 1] * 100:6.1f}%" for f in fracs)
        nz = s[s > 0]
        ratio = f"{nz[0] / nz[-1]:9.1f}" if nz.size else "        -"
        log(f"{il:5d}  {n:5d}  {cells}  {ratio}  {int((s == 0).sum()):5d}")


def is_per_expert(role: str) -> bool:
    # role is the tensor name without the "blk.N." prefix, e.g. "ffn_gate_exps.weight"
    return "_exps." in role or role.startswith("ffn_gate_inp.") or role.startswith("exp_probs_b")


def cmd_permute(args) -> None:
    counts = read_counts(args.imatrix)
    reader = gguf.GGUFReader(args.model)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    n_expert = int(reader.fields[f"{arch}.expert_count"].contents())

    orders: dict[int, np.ndarray] = {}
    for il, roles in counts.items():
        o = layer_order(roles)
        if o.size != n_expert:
            sys.exit(f"layer {il}: imatrix has {o.size} experts, model has {n_expert} - wrong imatrix for this model")
        orders[il] = o

    plan: dict[str, int] = {}  # tensor name -> layer whose order applies
    suspicious = []
    for t in reader.tensors:
        m = LAYER_RE.match(t.name)
        if not m:
            continue
        il, role = int(m.group(1)), m.group(2)
        if is_per_expert(role):
            if t.data.shape[0] != n_expert:
                sys.exit(f"{t.name}: outer dim {t.data.shape[0]} != n_expert {n_expert}")
            if il not in orders:
                sys.exit(f"{t.name}: per-expert tensor but the imatrix has no counts for layer {il}")
            plan[t.name] = il
        elif n_expert in t.data.shape:
            suspicious.append(t.name)
    missing = sorted(set(orders) - {il for il in plan.values()})
    if missing:
        sys.exit(f"imatrix has expert counts for layers {missing} but the model has no per-expert tensors there")
    if suspicious:
        # not fatal: n_expert can coincide with an ordinary dimension (head_dim 256, ...);
        # the logit-equality test is what proves nothing else is indexed by expert id
        log(f"note: {len(suspicious)} unpermuted tensors have a dim of size {n_expert}, e.g. {suspicious[:3]}")

    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} exists (use --force)")

    writer = gguf.GGUFWriter(args.out, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name.startswith("moe_tier."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub)
    writer.add_bool("moe_tier.permuted", True)
    writer.add_string("moe_tier.imatrix", os.path.basename(args.imatrix))

    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    total = sum(int(t.n_bytes) for t in reader.tensors)
    done = 0
    for t in reader.tensors:
        il = plan.get(t.name)
        data = t.data if il is None else np.ascontiguousarray(t.data[orders[il]])
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        done += int(t.n_bytes)
        if il is not None:
            print(f"\r{done / total * 100:5.1f}%  {t.name:<40}", end="", file=sys.stderr)
    writer.close()
    log(f"\nwrote {args.out}: {len(plan)} tensors permuted across {len(orders)} MoE layers")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stats", help="per-expert routing counts from an imatrix")
    s.add_argument("imatrix")
    s.set_defaults(fn=cmd_stats)
    s = sp.add_parser("permute", help="reorder experts hottest-first")
    s.add_argument("model")
    s.add_argument("imatrix")
    s.add_argument("out")
    s.add_argument("--force", action="store_true", help="overwrite OUT")
    s.set_defaults(fn=cmd_permute)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
