#!/bin/bash
# Capture a torch-profiler trace for one spec-decoding config under steady
# decode load, then write per-rank chrome traces to traces/<label>/.
#
#   TOPO=tp|pp  (default tp)   BLOCK=8|16 (default 8)   CONC=8  PORT=30011
#
# Example: TOPO=pp BLOCK=8 CONC=4 bash capture_profile.sh
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

TOPO=${TOPO:-tp}
BLOCK=${BLOCK:-8}
CONC=${CONC:-8}
PORT=${PORT:-30011}
STEPS=${STEPS:-50}
LABEL=${TOPO}2_dflash_b${BLOCK}_c${CONC}
OUT=traces/$LABEL
LOG=logs/${LABEL}_profile.log

PAR_FLAG=$([ "$TOPO" = pp ] && echo --pp-size || echo --tp-size)2

mkdir -p "$OUT" logs results

# 0) clean slate
pkill -f "sglang.launch_server" 2>/dev/null || true
sleep 8

# 1) launch server
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} nohup python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3.6-27B --trust-remote-code \
  $PAR_FLAG --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-27B-DFlash \
  --speculative-dflash-block-size $BLOCK \
  --mem-fraction-static 0.82 --cuda-graph-max-bs 32 --page-size 1 \
  --mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 2.0 \
  --enable-metrics --host 127.0.0.1 --port $PORT \
  > "$LOG" 2>&1 &
SERVER_PID=$!

# 2) wait healthy (up to 8 min)
for i in $(seq 1 60); do
  sleep 8
  curl -s -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
  ps -p $SERVER_PID >/dev/null || { echo "server died, see $LOG"; exit 1; }
done
curl -s -m 3 http://127.0.0.1:$PORT/health >/dev/null || { echo "health timeout"; exit 1; }
echo "[ok] server healthy (pid $SERVER_PID)"

# 3) drive steady decode load (8 concurrent x 2048 tokens)
nohup python3 bench_spectre.py \
  --url http://127.0.0.1:$PORT --label profload \
  --load-points "${CONC}:${CONC}:8" --max-tokens 2048 \
  --output-dir results >/dev/null 2>&1 &
LOAD_PID=$!

# 4) reach steady decode, then capture N decode steps (auto-stops)
sleep 25
curl -s http://127.0.0.1:$PORT/start_profile -H "Content-Type: application/json" -d '{
  "output_dir": "'"$SCRIPT_DIR/$OUT"'",
  "num_steps": '"$STEPS"',
  "activities": ["CPU", "GPU"]
}'
echo ""
echo "[ok] profiling $STEPS steps (auto-stops)..."

# 5) wait until trace files are fully written (size stable across 10s)
for i in $(seq 1 30); do
  sleep 10
  if ls $OUT/*.json.gz >/dev/null 2>&1; then
    S1=$(du -sb $OUT | cut -f1); sleep 10; S2=$(du -sb $OUT | cut -f1)
    [ "$S1" = "$S2" ] && [ "$S2" -gt 1000000 ] && break
  fi
done

# 6) teardown + post-process
kill $LOAD_PID 2>/dev/null || true
sleep 2
kill $SERVER_PID 2>/dev/null || true
sleep 8

python3 slim_trace.py $OUT/*trace.json.gz -o $OUT/lean.json.gz
echo "[done] raw + lean traces:"
ls -lh $OUT/
