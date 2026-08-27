#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

MODEL=${MODEL:-Qwen/Qwen3-8B}
DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}
MODEL_REVISION=${MODEL_REVISION:-main}
DRAFT_MODEL_REVISION=${DRAFT_MODEL_REVISION:-main}
DATASET=${DATASET:-$SCRIPT_DIR/data/sharegpt.json}
GPUS=${GPUS:-0,1}
NUM_GPUS=${NUM_GPUS:-}
MODES=${MODES:-tp,dp,pp_uniform,pp_auto}
UNIFORM_PARTITION=${UNIFORM_PARTITION:-18,18}
AUTO_PARTITION=${AUTO_PARTITION:-22,14}
ENABLE_REPLAY_SSM=${ENABLE_REPLAY_SSM:-0}
PORT=${PORT:-31000}
RAGGED_VERIFY_MODE=${RAGGED_VERIFY_MODE:-static}
LOAD_POINTS=${LOAD_POINTS:-8:2:32,16:4:64,32:8:128,64:16:256,96:24:384,128:32:512}
MAX_TOKENS=${MAX_TOKENS:-1024}
PROMPT_MAX_TOKENS=${PROMPT_MAX_TOKENS:-2000}
DFLASH_BLOCK_SIZE=${DFLASH_BLOCK_SIZE:-16}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-600}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-1800}
COOLDOWN_S=${COOLDOWN_S:-10}
MODEL_TAG=${MODEL//\//_}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRIPT_DIR/results/${MODEL_TAG}_$(date -u +%Y%m%d_%H%M%S)}

GPUS=${GPUS//[[:space:]]/}
IFS=',' read -r -a GPU_LIST <<<"$GPUS"
if ((${#GPU_LIST[@]} == 0)); then
  echo "GPUS must contain at least one GPU id" >&2
  exit 2
fi
for gpu in "${GPU_LIST[@]}"; do
  if [[ ! $gpu =~ ^[0-9]+$ ]]; then
    echo "GPUS must contain non-negative integer ids; got '$GPUS'" >&2
    exit 2
  fi
done
if [[ -z $NUM_GPUS ]]; then
  NUM_GPUS=${#GPU_LIST[@]}
fi
if [[ ! $NUM_GPUS =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer; got $NUM_GPUS" >&2
  exit 2
fi
if ((${#GPU_LIST[@]} != NUM_GPUS)); then
  echo "NUM_GPUS=$NUM_GPUS does not match GPUS=$GPUS (${#GPU_LIST[@]} ids)" >&2
  exit 2
fi
if [[ $ENABLE_REPLAY_SSM != 0 && $ENABLE_REPLAY_SSM != 1 ]]; then
  echo "ENABLE_REPLAY_SSM must be 0 or 1; got $ENABLE_REPLAY_SSM" >&2
  exit 2
fi
IFS=',' read -r -a raw_mode_list <<<"$MODES"
MODE_LIST=()
for raw_mode in "${raw_mode_list[@]}"; do
  mode=${raw_mode//[[:space:]]/}
  case "$mode" in
    tp | dp | pp_uniform | pp_auto) ;;
    *)
      echo "Unsupported mode '$raw_mode'; expected tp,dp,pp_uniform,pp_auto" >&2
      exit 2
      ;;
  esac
  already_selected=0
  for selected_mode in "${MODE_LIST[@]}"; do
    if [[ $selected_mode == "$mode" ]]; then
      already_selected=1
      break
    fi
  done
  if ((already_selected == 0)); then
    MODE_LIST+=("$mode")
  fi
done
if ((${#MODE_LIST[@]} == 0)); then
  echo "MODES must select at least one mode" >&2
  exit 2
fi

UNIFORM_PARTITION=${UNIFORM_PARTITION//[[:space:]]/}
AUTO_PARTITION=${AUTO_PARTITION//[[:space:]]/}
BASE_URL="http://127.0.0.1:$PORT"
SERVER_PID=

mode_enabled() {
  local expected=$1
  local mode
  for mode in "${MODE_LIST[@]}"; do
    [[ $mode == "$expected" ]] && return 0
  done
  return 1
}

validate_partition() {
  local name=$1
  local partition=$2
  local layer_count
  local -a layers
  IFS=',' read -r -a layers <<<"$partition"
  if ((${#layers[@]} != NUM_GPUS)); then
    echo "$name must contain $NUM_GPUS layer counts; got '$partition'" >&2
    exit 2
  fi
  for layer_count in "${layers[@]}"; do
    if [[ ! $layer_count =~ ^[1-9][0-9]*$ ]]; then
      echo "$name must contain positive integers; got '$partition'" >&2
      exit 2
    fi
  done
}

mode_enabled pp_uniform && validate_partition UNIFORM_PARTITION "$UNIFORM_PARTITION"
mode_enabled pp_auto && validate_partition AUTO_PARTITION "$AUTO_PARTITION"

cleanup() {
  if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT -- "-$SERVER_PID" 2>/dev/null || true
    for _ in {1..60}; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
      sleep 5
    fi
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local log_file=$1
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl -fsS "$BASE_URL/model_info" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited during startup; see $log_file" >&2
      tail -n 120 "$log_file" >&2 || true
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Server startup timed out; see $log_file" >&2
      tail -n 120 "$log_file" >&2 || true
      return 1
    fi
    sleep 1
  done
}

run_config() {
  local name=$1
  local tp_size=$2
  local pp_size=$3
  local dp_size=$4
  local partition=$5
  local load_point=$6
  local active_bs=$7
  local num_requests=$8
  local point_tag=$9
  shift 9
  local -a extra_server_args=("$@")
  local batch_divisor=1
  local divisor_name=world_size
  local cuda_graph_bs
  local max_running_requests=$active_bs
  local output_dir="$OUTPUT_ROOT/$name/$point_tag"
  local -a server_args

  if ((tp_size <= 0 || pp_size <= 0 || dp_size <= 0)); then
    echo "[$name] tp_size, pp_size, and dp_size must be positive" >&2
    exit 1
  fi
  if ((tp_size * pp_size * dp_size != NUM_GPUS)); then
    echo "[$name] TP${tp_size} x PP${pp_size} x DP${dp_size} must use NUM_GPUS=$NUM_GPUS" >&2
    exit 1
  fi
  if ((pp_size > 1 && dp_size > 1)); then
    echo "[$name] this runner supports pure DP or pure PP, not both together" >&2
    exit 1
  fi
  if ((dp_size > 1)); then
    batch_divisor=$dp_size
    divisor_name=dp_size
  elif ((pp_size > 1)); then
    batch_divisor=$pp_size
    divisor_name=pp_size
  fi
  if ((active_bs % batch_divisor != 0)); then
    echo "[$name] active_bs=$active_bs must be divisible by $divisor_name=$batch_divisor" >&2
    exit 1
  fi
  cuda_graph_bs=$((active_bs / batch_divisor))
  if ((dp_size > 1)); then
    max_running_requests=$cuda_graph_bs
  fi

  mkdir -p "$output_dir"
  server_args=(
    serve
    --model-path "$MODEL"
    --revision "$MODEL_REVISION"
    --tp-size "$tp_size"
    --pp-size "$pp_size"
    --dp-size "$dp_size"
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$DRAFT_MODEL"
    --speculative-draft-model-revision "$DRAFT_MODEL_REVISION"
    --speculative-draft-attention-backend flashinfer
    --speculative-dflash-block-size "$DFLASH_BLOCK_SIZE"
    --attention-backend flashinfer
    --max-running-requests "$max_running_requests"
    --load-balance-method round_robin
    --cuda-graph-max-bs-decode "$cuda_graph_bs"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --page-size 1
    --random-seed 1
    --disable-radix-cache
    --trust-remote-code
    --host 127.0.0.1
    --port "$PORT"
  )
  if [[ $ENABLE_REPLAY_SSM == 1 ]]; then
    server_args+=(
      --linear-attn-backend triton
      --mamba-ssm-dtype float32
      --mamba-full-memory-ratio 0.9
      --enable-linear-replayssm-spec
    )
  fi
  if [[ -n $partition ]]; then
    server_args+=(--pp-layer-partition "$partition")
  fi
  if ((pp_size > 1)); then
    server_args+=(--disable-overlap-schedule)
  fi
  server_args+=("${extra_server_args[@]}")

  echo "[$name][$load_point] starting: tp=$tp_size pp=$pp_size dp=$dp_size max_running=$max_running_requests cuda_graph_bs=$cuda_graph_bs replay_ssm=$ENABLE_REPLAY_SSM mem_fraction=$MEM_FRACTION_STATIC"
  CUDA_VISIBLE_DEVICES="$GPUS" \
    SGLANG_RAGGED_VERIFY_MODE="$RAGGED_VERIFY_MODE" \
    SGL_FORCE_SHUTDOWN=1 \
    setsid sglang "${server_args[@]}" >"$output_dir/server.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$output_dir/server.log"

  python "$SCRIPT_DIR/bench_spectre.py" \
    --url "$BASE_URL" \
    --label "${name}_r1" \
    --dataset "$DATASET" \
    --tokenizer "$MODEL" \
    --tokenizer-revision "$MODEL_REVISION" \
    --load-points "$load_point" \
    --max-tokens "$MAX_TOKENS" \
    --prompt-max-tokens "$PROMPT_MAX_TOKENS" \
    --temperature 0 \
    --seed 1 \
    --request-timeout-s "$REQUEST_TIMEOUT_S" \
    --cooldown-s "$COOLDOWN_S" \
    --output-dir "$output_dir"

  cleanup
  echo "[$name][$load_point] complete"
}

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"
echo "Results: $OUTPUT_ROOT"

IFS=',' read -r -a load_point_list <<<"$LOAD_POINTS"
for raw_load_point in "${load_point_list[@]}"; do
  load_point=${raw_load_point//[[:space:]]/}
  if [[ ! $load_point =~ ^([1-9][0-9]*):([0-9]+([.][0-9]*)?):([1-9][0-9]*)$ ]]; then
    echo "Invalid load point '$raw_load_point'; expected C:QPS:N" >&2
    exit 1
  fi

  active_bs=${BASH_REMATCH[1]}
  num_requests=${BASH_REMATCH[4]}
  point_tag="c${active_bs}_qps${BASH_REMATCH[2]}_n${num_requests}"

  if mode_enabled tp; then
    run_config "tp${NUM_GPUS}" "$NUM_GPUS" 1 1 "" \
      "$load_point" "$active_bs" "$num_requests" "$point_tag"
  fi
  if mode_enabled dp; then
    run_config "dp${NUM_GPUS}" 1 1 "$NUM_GPUS" "" \
      "$load_point" "$active_bs" "$num_requests" "$point_tag"
  fi
  if mode_enabled pp_uniform; then
    run_config "pp${NUM_GPUS}_uniform" 1 "$NUM_GPUS" 1 "$UNIFORM_PARTITION" \
      "$load_point" "$active_bs" "$num_requests" "$point_tag"
  fi
  if mode_enabled pp_auto; then
    run_config "pp${NUM_GPUS}_auto" 1 "$NUM_GPUS" 1 "$AUTO_PARTITION" \
      "$load_point" "$active_bs" "$num_requests" "$point_tag"
  fi
done

python "$SCRIPT_DIR/summarize_spectre.py" "$OUTPUT_ROOT"
python "$SCRIPT_DIR/plot_performance.py" "$OUTPUT_ROOT/summary.csv"
echo "All benchmark points complete: $OUTPUT_ROOT"
