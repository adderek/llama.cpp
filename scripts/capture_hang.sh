#!/usr/bin/env bash
# Snapshot a wedged llama-server. Run the moment repro reports HANG.
#   ./capture_hang.sh <pid>
# Needs: ptrace_scope=0 (sudo sysctl kernel.yama.ptrace_scope=0) for gdb attach
#        as same user; dmesg needs sudo. rocgdb (if installed) shows device state.
set -u
PID="${1:?usage: capture_hang.sh <pid>}"
TS="$(date +%Y%m%dT%H%M%S)"
OUT="hang_${PID}_${TS}"
mkdir -p "$OUT"

echo "[*] GPU state"; rocm-smi > "$OUT/rocm-smi.txt" 2>&1

DBG=gdb; command -v rocgdb >/dev/null && DBG=rocgdb
echo "[*] host stacks via $DBG"
"$DBG" -p "$PID" -batch \
    -ex "info threads" \
    -ex "thread apply all bt" > "$OUT/${DBG}-bt.txt" 2>&1

echo "[*] kernel log (needs sudo; skipped if no rights)"
sudo -n dmesg 2>/dev/null | grep -iE 'amdgpu|ring.*timeout|VM_L2|gpu reset|kfd' \
    > "$OUT/dmesg-gpu.txt" 2>&1 || echo "dmesg: no sudo rights" > "$OUT/dmesg-gpu.txt"

echo "[*] per-thread CPU (1s delta) - 0 ticks = fully blocked"
for t in /proc/$PID/task/*; do
    awk '{print FILENAME, $14, $15}' "$t/stat" 2>/dev/null
done > "$OUT/cpu.txt"

echo "saved -> $OUT/"
