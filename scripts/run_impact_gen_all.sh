#!/bin/bash
# Poll for precompute supply caches and launch per-region impact dataset generation.
CACHE_DIR=/tmp/scenario_cache
OUT_DIR=data/aemo_dt_impact
LOG_DIR=/tmp/impact_gen_logs
mkdir -p "$LOG_DIR"
REPO=/media/victoru/0a1c0748-f508-de49-9b25-b0ac435a9727/energydecision
cd "$REPO"

REGIONS="NSW1 QLD1 SA1 TAS1 VIC1"
declare -A EPISODES=( [NSW1]=250 [QLD1]=250 [SA1]=200 [TAS1]=250 [VIC1]=200 )

echo "[orchestrator] started $(date)"
while true; do
    for R in $REGIONS; do
        PKL="$CACHE_DIR/${R}_supply.pkl"
        DONE_MARK="$LOG_DIR/${R}.done"
        LOCK="$LOG_DIR/${R}.lock"
        if [ -f "$PKL" ] && [ ! -f "$DONE_MARK" ] && [ ! -f "$LOCK" ]; then
            # SERIALIZE: only one region generation at a time (each spawns 6
            # workers that load the v2 DT model ~1.4GB; concurrent regions
            # over-subscribe the 22GB VRAM and OOM).
            RUNNING=$(ps -eo cmd | grep "generate_impact_dataset.py --regions" | grep -v grep | wc -l)
            if [ "$RUNNING" -gt 0 ]; then
                echo "[orchestrator] $R ready but generation busy ($RUNNING running); waiting"
                continue
            fi
            touch "$LOCK"
            N=${EPISODES[$R]}
            echo "[orchestrator] launching $R generation ($N eps) at $(date)"
            setsid nohup distrobox enter energydecision-gpu -- python3 -u \
                scripts/generate_impact_dataset.py --regions "$R" --n-episodes "$N" \
                --workers 6 --out "$OUT_DIR" \
                > "$LOG_DIR/${R}.log" 2>&1 < /dev/null &
            disown
        fi
        # Mark done when its log shows the "Wrote ... episodes" line
        if [ -f "$LOG_DIR/${R}.log" ] && grep -q "Wrote .* episodes" "$LOG_DIR/${R}.log"; then
            touch "$DONE_MARK"
            echo "[orchestrator] $R generation complete $(date)"
        fi
    done
    # Stop when all done
    DONE_COUNT=0
    for R in $REGIONS; do
        [ -f "$LOG_DIR/${R}.done" ] && DONE_COUNT=$((DONE_COUNT+1))
    done
    if [ "$DONE_COUNT" -eq 5 ]; then
        echo "[orchestrator] all 5 regions generated $(date)"
        break
    fi
    sleep 120
done
