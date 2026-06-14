#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-full_learning_baseline_$(date +%Y%m%d_%H%M%S)}"
LOGDIR="system_logs/${TAG}"
mkdir -p "$LOGDIR"

MON_SECS="${MON_SECS:-86400}"  # default: 24h

SIGNAL=""
TRAIN_EXIT=0

finalize() {
  # Always run, even if the training command fails or the shell is hung up.
  local ec="$?"
  set +e

  echo "end $(date -Is)" >> "$LOGDIR/marker.txt" 2>/dev/null || true
  echo "exit_code=$ec" > "$LOGDIR/exit_code.txt" 2>/dev/null || true
  [ -n "${SIGNAL}" ] && echo "signal=${SIGNAL}" > "$LOGDIR/signal.txt" 2>/dev/null || true
  echo "train_exit=$TRAIN_EXIT" > "$LOGDIR/train_exit.txt" 2>/dev/null || true

  nvidia-smi -L > "$LOGDIR/nvidia_smi_list.txt" 2>&1 || true
  nvidia-smi -q > "$LOGDIR/nvidia_smi_q.txt" 2>&1 || true

  journalctl -k --since "$(date -Is --date='60 minutes ago')" --no-pager | tail -n 3000 > "$LOGDIR/journalctl_k_tail.txt" 2>&1 || true
  journalctl -k -b -1 --no-pager | tail -n 3000 > "$LOGDIR/journalctl_k_prevboot_tail.txt" 2>&1 || true
  [ -f /var/log/kern.log ] && tail -n 3000 /var/log/kern.log > "$LOGDIR/kernlog_tail.txt" 2>&1 || true

  local crash=0

  # Best-effort progress summary (useful even when the training crashes)
  (
    PROG_JSON="models/aemo/dt/${TAG}/learning_baseline/aemo_dt_loss_history_progress.json"
    MON_LOG="models/aemo/dt/${TAG}/learning_baseline/aemo_dt_loss_history_monitor.log"

    {
      echo "training_progress_summary:"
      if [ -f "$PROG_JSON" ]; then
        python3 - <<PY
import json
from pathlib import Path
p=Path("$PROG_JSON")
obj=json.loads(p.read_text())
print(f"- epoch: {obj.get('epoch')}/{obj.get('epochs')}")
print(f"- segment: {obj.get('segment')} progress_fraction={obj.get('progress_fraction')}")
ct=obj.get('current_train',{})
print(f"- current_train.batch_idx: {ct.get('batch_idx')}")
best=obj.get('best',{})
print(f"- best.val_total: {best.get('score')}")
print(f"- snapshot_timestamp: {obj.get('timestamp')}")
PY
      else
        echo "- progress json not found: $PROG_JSON"
      fi

      if [ -f "$MON_LOG" ]; then
        echo "- last_epoch_line:"
        egrep -a "Epoch [0-9]+/" "$MON_LOG" | tail -n 1
        echo "- last_error_lines:"
        egrep -a -n "CUDA error:|unspecified launch failure|c10::AcceleratorError|SIGABRT|Xid|GPU is lost" "$MON_LOG" | tail -n 10
      else
        echo "- monitor log not found: $MON_LOG"
      fi
    } > "$LOGDIR/training_progress.txt"
  ) || true

  # Kernel evidence (Xid/GPU lost)
  if tail -n 5000 "$LOGDIR/journalctl_k_follow.txt" 2>/dev/null | grep -E "Xid 79|Xid \(.*\): 79|GPU is lost|NVRM" >/dev/null 2>&1; then
    crash=1
  fi

  # nvidia-smi evidence
  if tail -n 2000 "$LOGDIR/nvidia_smi_query.csv" 2>/dev/null | grep -F "[GPU is lost]" >/dev/null 2>&1; then
    crash=1
  fi

  # PyTorch/CUDA evidence (from trainer logs if present)
  if grep -E "CUDA error:|unspecified launch failure|c10::AcceleratorError" -n "models/aemo/dt/${TAG}/learning_baseline/aemo_dt_loss_history_monitor.log" >/dev/null 2>&1; then
    crash=1
  fi

  # Non-zero exits count as a crash unless monitors say otherwise.
  if [ "$ec" -ne 0 ] || [ "$TRAIN_EXIT" -ne 0 ]; then
    crash=1
  fi

  if [ "$crash" -eq 1 ]; then
    echo "Detected a GPU/CUDA crash signature (Xid/GPU-lost/CUDA error or non-zero exit)." > "$LOGDIR/CRASH_DETECTED.txt"
    {
      echo "Telemetry captured under $LOGDIR"
      echo "GPU appears unstable (Xid/GPU lost / CUDA failure). System shutdown/reboot is REQUIRED after logs are saved."
      [ -f "$LOGDIR/training_progress.txt" ] && cat "$LOGDIR/training_progress.txt"
    } > "$LOGDIR/SAFE_TO_SHUTDOWN.txt"
  else
    {
      echo "Telemetry captured under $LOGDIR"
      echo "Run finished without detecting Xid79/GPU-lost in the captured telemetry."
      echo "It is safe to exit Distrobox and shut down/reboot once you have this directory."
      [ -f "$LOGDIR/training_progress.txt" ] && cat "$LOGDIR/training_progress.txt"
    } > "$LOGDIR/SAFE_TO_SHUTDOWN.txt"
  fi
}

on_signal() {
  SIGNAL="$1"
  echo "signal $SIGNAL $(date -Is)" >> "$LOGDIR/marker.txt" 2>/dev/null || true
  exit 128
}

trap finalize EXIT
trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

echo "start $(date -Is)" > "$LOGDIR/marker.txt"
echo "Starting host-side telemetry into: $LOGDIR"

# (1) periodic point-in-time stats (includes PCIe link gen/width)
(timeout "${MON_SECS}s" nvidia-smi --query-gpu=timestamp,pcie.link.gen.gpucurrent,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max,temperature.gpu,utilization.gpu,utilization.memory,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.total \
  --format=csv -l 2 > "$LOGDIR/nvidia_smi_query.csv" 2>&1 || true) &
echo $! > "$LOGDIR/nvidia_smi_query.pid"

# (2) streaming dmon (includes PCIe throughput 't' and PCIe replay/ECC errors 'e')
(timeout "${MON_SECS}s" nvidia-smi dmon -s putec -d 2 > "$LOGDIR/nvidia_dmon.txt" 2>&1 || true) &
echo $! > "$LOGDIR/nvidia_dmon.pid"

# (3) cpu + disk (vmstat + iostat)
(timeout "${MON_SECS}s" bash -lc '
  i=0
  while true; do
    date -Is >> "'$LOGDIR'/timeline.txt"
    vmstat 1 2 | tail -n 1 >> "'$LOGDIR'/vmstat.txt" 2>/dev/null || true
    if [ $((i % 5)) -eq 0 ]; then
      iostat -dx 1 1 >> "'$LOGDIR'/iostat.txt" 2>/dev/null || true
    fi
    i=$((i+1))
    sleep 1
  done
' > "$LOGDIR/monitor_stdout_stderr.log" 2>&1 || true) &

# (4) continuous kernel log capture (important for Xid79 timeline)
(timeout "${MON_SECS}s" journalctl -k -f --no-pager > "$LOGDIR/journalctl_k_follow.txt" 2>&1 || true) &

# (optional) quick CUDA smoke check while telemetry is already running
/usr/bin/distrobox-enter energydecision-gpu -- python3 - <<'PY' > "$LOGDIR/torch_healthcheck.txt" 2>&1 || true
import time
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())

if torch.cuda.is_available():
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    print("matmul_s", round(time.time() - t0, 3), "sum", float(c.sum().item()))
PY

# ---- training command (inside the GPU distrobox) ----
RUN_TAG="$TAG"
echo "run_tag=$RUN_TAG" > "$LOGDIR/run_tag.txt"
echo "Launching learning-baseline..." | tee "$LOGDIR/launch_marker.txt"

set +e
/usr/bin/distrobox-enter energydecision-gpu -- python3 src/launch_aemo_training.py \
  --run-tier learning-baseline \
  --runtime-mode allow-host \
  --run-tag "$RUN_TAG" \
  --optimizer adamw \
  --scheduler steplr \
  --n-block 8 \
  --h-dim 512 \
  --n-heads 8 \
  --drop-p 0.15 \
  --context-length 180 \
  > "$LOGDIR/train_stdout_stderr.log" 2>&1
TRAIN_EXIT=$?
set -e

# Preserve the exit status for callers while still letting the EXIT trap write markers.
exit "$TRAIN_EXIT"
