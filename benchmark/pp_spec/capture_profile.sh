#!/usr/bin/env bash
# Low-level trace capture helper. For automatic PP partition tuning, use:
#   python adaptive_pp_tuner.py --pp-size 2 --tp-size 1
#
# Compatibility defaults:
#   TOPO=tp -> TP_SIZE=2 PP_SIZE=1
#   TOPO=pp -> TP_SIZE=1 PP_SIZE=2
#
# General hybrid example:
#   TP_SIZE=2 PP_SIZE=4 CONC=32 bash capture_profile.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$SCRIPT_DIR"

TOPO=${TOPO:-tp}
TP_SIZE=${TP_SIZE:-}
PP_SIZE=${PP_SIZE:-}
if [ -z "$TP_SIZE" ] && [ -z "$PP_SIZE" ]; then
  if [ "$TOPO" = pp ]; then
    TP_SIZE=1
    PP_SIZE=2
  elif [ "$TOPO" = tp ]; then
    TP_SIZE=2
    PP_SIZE=1
  else
    echo "TOPO must be tp or pp" >&2
    exit 2
  fi
else
  TP_SIZE=${TP_SIZE:-1}
  PP_SIZE=${PP_SIZE:-1}
fi

if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ && "$PP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE and PP_SIZE must be positive integers" >&2
  exit 2
fi

BLOCK=${BLOCK:-8}
CONC=${CONC:-32}
if ! [[ "$CONC" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONC must be a positive integer" >&2
  exit 2
fi
PORT=${PORT:-30011}
STEPS=${STEPS:-50}
NUM_LAYERS=${NUM_LAYERS:-64}
RANDOM_SEED=${RANDOM_SEED:-1}
PROMPT_TOKENS=${PROMPT_TOKENS:-256}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-}
PP_PARTITION=${PP_PARTITION:-${SGLANG_PP_LAYER_PARTITION:-}}
PP_MAX_LAYERS=${PP_MAX_LAYERS:-}
WORLD_SIZE=$((TP_SIZE * PP_SIZE))

ACTIVE_SUFFIX=
MAX_RUNNING_ARGS=()
if [ -n "$MAX_RUNNING_REQUESTS" ]; then
  if ! [[ "$MAX_RUNNING_REQUESTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_RUNNING_REQUESTS must be a positive integer" >&2
    exit 2
  fi
  if [ "$MAX_RUNNING_REQUESTS" -gt "$CONC" ]; then
    echo "MAX_RUNNING_REQUESTS cannot exceed CONC=$CONC" >&2
    exit 2
  fi
  MAX_RUNNING_ARGS=(--max-running-requests "$MAX_RUNNING_REQUESTS")
  ACTIVE_SUFFIX=_a${MAX_RUNNING_REQUESTS}
fi

PARTITION_SUFFIX=
if [ -n "$PP_PARTITION" ]; then
  if [ "$PP_SIZE" -le 1 ]; then
    echo "PP_PARTITION requires PP_SIZE > 1" >&2
    exit 2
  fi
  if [[ ! "$PP_PARTITION" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "PP_PARTITION must be comma-separated positive layer counts" >&2
    exit 2
  fi
  IFS=, read -r -a PARTS <<< "$PP_PARTITION"
  if [ "${#PARTS[@]}" -ne "$PP_SIZE" ]; then
    echo "PP_PARTITION has ${#PARTS[@]} entries, expected PP_SIZE=$PP_SIZE" >&2
    exit 2
  fi
  PARTITION_SUM=0
  for count in "${PARTS[@]}"; do
    PARTITION_SUM=$((PARTITION_SUM + count))
  done
  if [ "$PARTITION_SUM" -ne "$NUM_LAYERS" ]; then
    echo "PP_PARTITION=$PP_PARTITION does not sum to NUM_LAYERS=$NUM_LAYERS" >&2
    exit 2
  fi
  export SGLANG_PP_LAYER_PARTITION="$PP_PARTITION"
  PARTITION_SUFFIX=_p${PP_PARTITION//,/-}
else
  unset SGLANG_PP_LAYER_PARTITION
fi

LABEL=tp${TP_SIZE}_pp${PP_SIZE}_dflash_b${BLOCK}_c${CONC}${ACTIVE_SUFFIX}${PARTITION_SUFFIX}
OUT=traces/$LABEL
LOG=logs/${LABEL}_profile.log
mkdir -p "$OUT" logs results

SGLANG_BIN=${SGLANG_BIN:-}
if [ -z "$SGLANG_BIN" ] && [ -x "$REPO_ROOT/.venv/bin/sglang" ]; then
  SGLANG_BIN=$REPO_ROOT/.venv/bin/sglang
elif [ -z "$SGLANG_BIN" ]; then
  SGLANG_BIN=$(command -v sglang 2>/dev/null || true)
fi
if [ ! -x "$SGLANG_BIN" ]; then
  echo "sglang executable not found; set SGLANG_BIN=/path/to/sglang" >&2
  exit 2
fi

PYTHON_BIN=${PYTHON_BIN:-}
if [ -z "$PYTHON_BIN" ] && [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN=$REPO_ROOT/.venv/bin/python
elif [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN=$(command -v python3 2>/dev/null || true)
fi
if [ ! -x "$PYTHON_BIN" ]; then
  echo "python executable not found; set PYTHON_BIN=/path/to/python" >&2
  exit 2
fi

SERVER_PID=
LOAD_PID=
stop_group() {
  local pid=${1:-}
  if [ -z "$pid" ]; then
    return
  fi
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
}
cleanup() {
  stop_group "$LOAD_PID" 2>/dev/null
  stop_group "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy server; choose another PORT" >&2
  exit 2
fi

HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
  SGLANG_FORCE_SHUTDOWN=${SGLANG_FORCE_SHUTDOWN:-1} \
  setsid "$SGLANG_BIN" serve \
  --model-path Qwen/Qwen3.6-27B --trust-remote-code \
  --tp-size "$TP_SIZE" --pp-size "$PP_SIZE" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path z-lab/Qwen3.6-27B-DFlash \
  --speculative-dflash-block-size "$BLOCK" \
  --mem-fraction-static 0.82 --cuda-graph-max-bs-decode 32 --page-size 1 \
  --random-seed "$RANDOM_SEED" \
  --mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 2.0 \
  "${MAX_RUNNING_ARGS[@]}" \
  --enable-metrics --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
  sleep 8
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server died, see $LOG" >&2
    exit 1
  fi
done
curl -fsS -m 3 "http://127.0.0.1:$PORT/health" >/dev/null || {
  echo "health timeout, see $LOG" >&2
  exit 1
}
if [ -n "$MAX_RUNNING_REQUESTS" ]; then
  mapfile -t RESOLVED_ACTIVE_CAPS < <(
    sed -n 's/.*max_total_num_tokens=[0-9][0-9]*.*max_running_requests=\([0-9][0-9]*\).*/\1/p' "$LOG"
  )
  if [ "${#RESOLVED_ACTIVE_CAPS[@]}" -eq 0 ]; then
    echo "could not resolve max_running_requests from $LOG" >&2
    exit 1
  fi
  for resolved_cap in "${RESOLVED_ACTIVE_CAPS[@]}"; do
    if [ "$resolved_cap" -ne "$MAX_RUNNING_REQUESTS" ]; then
      echo "requested MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS but server resolved $resolved_cap; see $LOG" >&2
      exit 1
    fi
  done
  echo "[ok] server healthy (pid $SERVER_PID, active cap $MAX_RUNNING_REQUESTS)"
else
  echo "[ok] server healthy (pid $SERVER_PID)"
fi

setsid "$PYTHON_BIN" bench_spectre.py \
  --url "http://127.0.0.1:$PORT" --label profload \
  --synthetic --load-points "${CONC}:1000:${CONC}" --max-tokens 4096 \
  --prompt-max-tokens "$PROMPT_TOKENS" \
  --cooldown-s 0 --output-dir results > /dev/null 2>&1 &
LOAD_PID=$!

sleep 25
if ! kill -0 "$LOAD_PID" 2>/dev/null; then
  echo "profile load exited before capture" >&2
  exit 1
fi
PROFILE_MARKER=$(mktemp "$OUT/.profile_started.XXXXXX")
curl -fsS "http://127.0.0.1:$PORT/start_profile" \
  -H "Content-Type: application/json" -d '{
    "output_dir": "'"$SCRIPT_DIR/$OUT"'",
    "num_steps": '"$STEPS"',
    "activities": ["CPU", "GPU"],
    "with_stack": false,
    "record_shapes": false
  }'
echo
echo "[ok] profiling $STEPS steps (auto-stops)..."

for _ in $(seq 1 30); do
  sleep 10
  mapfile -t CURRENT_TRACES < <(
    find "$OUT" -maxdepth 1 -type f -name '*trace.json.gz' \
      -newer "$PROFILE_MARKER" -print | sort
  )
  if [ "${#CURRENT_TRACES[@]}" -eq "$WORLD_SIZE" ]; then
    SIZE1=$(du -cb "${CURRENT_TRACES[@]}" | tail -n 1 | cut -f1)
    sleep 10
    SIZE2=$(du -cb "${CURRENT_TRACES[@]}" | tail -n 1 | cut -f1)
    if [ "$SIZE1" = "$SIZE2" ] && [ "$SIZE2" -gt 100000 ]; then
      break
    fi
  fi
done

stop_group "$LOAD_PID" 2>/dev/null
LOAD_PID=
stop_group "$SERVER_PID" 2>/dev/null
SERVER_PID=

mapfile -t TRACE_FILES < <(
  find "$OUT" -maxdepth 1 -type f -name '*trace.json.gz' \
    -newer "$PROFILE_MARKER" -print | sort
)
rm -f "$PROFILE_MARKER"
if [ "${#TRACE_FILES[@]}" -ne "$WORLD_SIZE" ]; then
  echo "expected $WORLD_SIZE rank traces in $OUT, found ${#TRACE_FILES[@]}" >&2
  exit 1
fi

"$PYTHON_BIN" slim_trace.py "${TRACE_FILES[@]}" -o "$OUT/lean.json.gz"

if [ "$PP_SIZE" -gt 1 ]; then
  ANALYZE_ARGS=(--num-layers "$NUM_LAYERS" --json-output "$OUT/partition.json")
  if [ -n "$PP_PARTITION" ]; then
    ANALYZE_ARGS+=(--current-partition "$PP_PARTITION")
  fi
  if [ -n "$PP_MAX_LAYERS" ]; then
    ANALYZE_ARGS+=(--max-layers-per-stage "$PP_MAX_LAYERS")
  fi
  "$PYTHON_BIN" auto_pp_partition.py "${TRACE_FILES[@]}" "${ANALYZE_ARGS[@]}" \
    | tee "$OUT/partition.txt"
fi

echo "[done] raw + lean traces:"
ls -lh "$OUT"/
