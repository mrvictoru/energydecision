#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-proxy_gpu_test_$(date +%Y%m%d_%H%M%S)}"
LOGDIR="system_logs/${TAG}"
mkdir -p "$LOGDIR"

echo "Monitoring dir: $LOGDIR"
echo "GPU stress test + pipeline smoke test"
echo "Tag: $TAG"

# ---- Background telemetry ----
# (1) nvidia-smi periodic stats
nvidia-smi --query-gpu=timestamp,pcie.link.gen.gpucurrent,pcie.link.width.current,temperature.gpu,utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem,memory.used,memory.total,ecc.errors.volatile.dram,ecc.errors.volatile.sram \
  --format=csv -l 2 > "$LOGDIR/nvidia_smi_query.csv" 2>&1 &
echo $! > "$LOGDIR/nvidia_smi_query.pid"

# (2) nvidia dmon (PCIe throughput, replay)
nvidia-smi dmon -s putc -d 2 > "$LOGDIR/nvidia_dmon.txt" 2>&1 &
echo $! > "$LOGDIR/nvidia_dmon.pid"

# (3) kernel log
journalctl -k -f --no-pager > "$LOGDIR/journalctl_k_follow.txt" 2>&1 &
JOURNAL_PID=$!
echo $JOURNAL_PID > "$LOGDIR/journalctl_k_follow.pid"

# (4) vmstat + iostat
(
  i=0
  while true; do
    date -Is >> "$LOGDIR/timeline.txt"
    vmstat 1 2 | tail -n 1 >> "$LOGDIR/vmstat.txt" 2>/dev/null || true
    if [ $((i % 5)) -eq 0 ]; then
      iostat -dx 1 1 >> "$LOGDIR/iostat.txt" 2>/dev/null || true
    fi
    i=$((i+1))
    sleep 1
  done
) > "$LOGDIR/monitor_stdout_stderr.log" 2>&1 &
echo $! > "$LOGDIR/monitor_subsystem.pid"

echo "Telemetry started. PIDs: $(cat "$LOGDIR"/*.pid 2>/dev/null | tr '\n' ' ')"

# ---- Pre-test CUDA health check ----
python3 -c "
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  [{i}] {p.name}, {p.total_memory/1024**3:.1f}GB, compute {p.major}.{p.minor}')
" | tee "$LOGDIR/torch_healthcheck.txt"

# ---- Launch training with dashboard ----
echo "=== Launching proxy-baseline training with dt_progress_runner dashboard ==="
set +e
python3 src/launch_aemo_training.py \
  --run-tier proxy-baseline \
  --runtime-mode allow-host \
  --run-tag "$TAG"
TRAIN_EXIT=$?
set -e

echo "Training exit code: $TRAIN_EXIT" | tee "$LOGDIR/training_exit.txt"

# ---- Stop telemetry ----
for pidfile in "$LOGDIR"/*.pid; do
  [ -f "$pidfile" ] && kill $(cat "$pidfile") 2>/dev/null || true
done
sleep 2
echo "Telemetry stopped."

# ---- Post-test kernel log capture ----
journalctl -k --since "$(date -Is --date='5 minutes ago')" --no-pager | tail -n 2000 > "$LOGDIR/journalctl_k_tail.txt" 2>&1 || true

# ---- Check for Xid/GPU-lost ----
CRASH=0
if grep -E "Xid 79|NVRM|GPU is lost" "$LOGDIR/journalctl_k_follow.txt" 2>/dev/null; then
  CRASH=1
  echo "Xid79/GPU-lost detected in kernel logs!"
fi
if grep -F "[GPU is lost]" "$LOGDIR/nvidia_smi_query.csv" 2>/dev/null; then
  CRASH=1
  echo "GPU-lost detected in nvidia-smi!"
fi
if grep -E "CUDA error:|AcceleratorError" "$LOGDIR/../$TAG"* 2>/dev/null; then
  CRASH=1
  echo "CUDA error detected in training logs!"
fi

if [ "$CRASH" -eq 1 ]; then
  echo "CRASH DETECTED — Xid79 or GPU lost. See $LOGDIR" | tee "$LOGDIR/CRASH_DETECTED.txt"
else
  echo "No GPU crash detected." | tee "$LOGDIR/SAFE_TO_SHUTDOWN.txt"
fi

# ---- Summary ----
echo ""
echo "=== Summary ==="
echo "Training exit code: $TRAIN_EXIT"
echo "Crash detected: $CRASH"
echo "Logs: $LOGDIR"
echo "Training run tag: $TAG"
echo "Model: models/aemo/dt/$TAG/proxy_baseline/"
