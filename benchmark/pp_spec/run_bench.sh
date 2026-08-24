#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# MODEL=${MODEL:-Qwen/Qwen3.5-9B}
# DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3.5-9B-DFlash}
MODEL=${MODEL:-Qwen/Qwen3-8B}
DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}
DATASET=${DATASET:-$SCRIPT_DIR/data/sharegpt.json}
GPUS=${GPUS:-0,1}
PORT=${PORT:-31000}
REPEATS=${REPEATS:-1}
RAGGED_VERIFY_MODE=${RAGGED_VERIFY_MODE:-static}
LOAD_POINTS=${LOAD_POINTS:-4:1000:16,8:1000:32,16:1000:64,32:1000:128,64:1000:256,96:1000:384,128:1000:512}
MAX_TOKENS=${MAX_TOKENS:-1024}
PROMPT_MAX_TOKENS=${PROMPT_MAX_TOKENS:-2000}
DFLASH_BLOCK_SIZE=${DFLASH_BLOCK_SIZE:-16}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-600}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-1800}
COOLDOWN_S=${COOLDOWN_S:-10}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRIPT_DIR/results/replayssm_$(date -u +%Y%m%d_%H%M%S)}

BASE_URL="http://127.0.0.1:$PORT"
SERVER_PID=

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

check_gpus_idle() {
  local gpu
  local processes
  IFS=',' read -r -a gpu_list <<<"$GPUS"
  if ((${#gpu_list[@]} != 2)); then
    echo "This benchmark expects exactly two GPU ids; got GPUS=$GPUS" >&2
    exit 1
  fi
  for gpu in "${gpu_list[@]}"; do
    processes=$(nvidia-smi -i "$gpu" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true)
    if [[ -n $processes ]]; then
      echo "GPU $gpu is occupied; refusing to disturb another workload:" >&2
      echo "$processes" >&2
      exit 1
    fi
  done
}

repeat_complete() {
  local summary_file=$1
  local active_bs=$2
  local num_requests=$3
  [[ -f $summary_file ]] || return 1
  jq -e \
    --argjson active_bs "$active_bs" \
    --argjson num_requests "$num_requests" \
    'length == 1
      and .[0].max_concurrency == $active_bs
      and .[0].num_requests == $num_requests
      and .[0].completed == $num_requests' \
    "$summary_file" >/dev/null
}

config_complete() {
  local output_dir=$1
  local name=$2
  local active_bs=$3
  local num_requests=$4
  local repeat
  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    repeat_complete \
      "$output_dir/${name}_r${repeat}_summary.json" \
      "$active_bs" \
      "$num_requests" || return 1
  done
}

run_config() {
  local name=$1
  local tp_size=$2
  local pp_size=$3
  local partition=$4
  local load_point=$5
  local active_bs=$6
  local num_requests=$7
  local point_tag=$8
  shift 8
  local -a extra_server_args=("$@")
  local cuda_graph_bs=$((active_bs / pp_size))
  local pp_micro_batch_size=$cuda_graph_bs
  local output_dir="$OUTPUT_ROOT/$name/$point_tag"
  local repeat
  local -a server_args

  if ((active_bs % pp_size != 0)); then
    echo "[$name] active_bs=$active_bs must be divisible by pp_size=$pp_size" >&2
    exit 1
  fi
  if config_complete "$output_dir" "$name" "$active_bs" "$num_requests"; then
    echo "[$name][$load_point] already complete; skipping"
    return
  fi
  mkdir -p "$output_dir"

  if [[ -n $partition ]]; then
    export SGLANG_PP_LAYER_PARTITION=$partition
  else
    unset SGLANG_PP_LAYER_PARTITION
  fi

  server_args=(
    serve
    --model-path "$MODEL"
    --tp-size "$tp_size"
    --pp-size "$pp_size"
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$DRAFT_MODEL"
    --speculative-draft-attention-backend flashinfer
    --speculative-dflash-block-size "$DFLASH_BLOCK_SIZE"
    --attention-backend flashinfer
    # --linear-attn-backend triton
    # --enable-linear-replayssm-spec
    --max-running-requests "$active_bs"
    --cuda-graph-max-bs-decode "$cuda_graph_bs"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --page-size 1
    --random-seed 1
    # --mamba-ssm-dtype bfloat16
    # --mamba-full-memory-ratio 2.0
    --disable-radix-cache
    --trust-remote-code
    --host 127.0.0.1
    --port "$PORT"
  )
  server_args+=("${extra_server_args[@]}")

  check_gpus_idle
  echo "[$name][$load_point] starting: tp=$tp_size pp=$pp_size pp_micro_batch_size=$pp_micro_batch_size mem_fraction=$MEM_FRACTION_STATIC extra_args='${extra_server_args[*]}'"
  CUDA_VISIBLE_DEVICES="$GPUS" \
    SGLANG_RAGGED_VERIFY_MODE="$RAGGED_VERIFY_MODE" \
    SGLANG_FORCE_SHUTDOWN=1 \
    setsid sglang "${server_args[@]}" >"$output_dir/server.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$output_dir/server.log"

  for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    if repeat_complete \
      "$output_dir/${name}_r${repeat}_summary.json" \
      "$active_bs" \
      "$num_requests"; then
      echo "[$name][$load_point][repeat=$repeat] already complete; skipping"
      continue
    fi
    python "$SCRIPT_DIR/bench_spectre.py" \
      --url "$BASE_URL" \
      --label "${name}_r${repeat}" \
      --dataset "$DATASET" \
      --tokenizer "$MODEL" \
      --load-points "$load_point" \
      --max-tokens "$MAX_TOKENS" \
      --prompt-max-tokens "$PROMPT_MAX_TOKENS" \
      --temperature 0 \
      --seed 1 \
      --request-timeout-s "$REQUEST_TIMEOUT_S" \
      --cooldown-s "$COOLDOWN_S" \
      --output-dir "$output_dir"
  done

  cleanup
  echo "[$name][$load_point] complete"
}

[[ -f $DATASET ]] || {
  echo "ShareGPT dataset not found: $DATASET" >&2
  exit 1
}
for command in sglang curl jq setsid nvidia-smi; do
  command -v "$command" >/dev/null || {
    echo "Required command not found: $command" >&2
    exit 1
  }
done

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"
echo "Results: $OUTPUT_ROOT"
echo "replayssm_spec=1 ragged_verify_mode=$RAGGED_VERIFY_MODE"
if [[ ! -f $OUTPUT_ROOT/manifest.txt ]]; then
  {
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_head=$(git rev-parse HEAD)"
    echo "model=$MODEL"
    echo "draft_model=$DRAFT_MODEL"
    echo "dataset=$DATASET"
    echo "ragged_verify_mode=$RAGGED_VERIFY_MODE"
    echo "load_points=$LOAD_POINTS"
    echo "max_tokens=$MAX_TOKENS"
    echo "prompt_max_tokens=$PROMPT_MAX_TOKENS"
    echo "mem_fraction_static=$MEM_FRACTION_STATIC"
    echo "workspace_status_begin"
    git status --short
    echo "workspace_status_end"
    sha256sum \
      "$DATASET" \
      "$SCRIPT_DIR/bench_spectre.py" \
      "$SCRIPT_DIR/run_bench.sh" \
      "$REPO_ROOT/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py" \
      "$REPO_ROOT/python/sglang/srt/mem_cache/kv_cache_configurator.py"
  } >"$OUTPUT_ROOT/manifest.txt"
fi

IFS=',' read -r -a load_point_list <<<"$LOAD_POINTS"
for raw_load_point in "${load_point_list[@]}"; do
  load_point=${raw_load_point//[[:space:]]/}
  if [[ ! $load_point =~ ^([1-9][0-9]*):([0-9]+([.][0-9]*)?):([1-9][0-9]*)$ ]]; then
    echo "Invalid load point '$raw_load_point'; expected C:QPS:N" >&2
    exit 1
  fi

  active_bs=${BASH_REMATCH[1]}
  request_rate=${BASH_REMATCH[2]}
  num_requests=${BASH_REMATCH[4]}
  point_tag="c${active_bs}_qps${request_rate}_n${num_requests}"

  # -----------------------------------------------------------------------
  # Experiment matrix: edit only these run_config lines.
  # Extra arguments after "$point_tag" are appended directly to `sglang serve`.
  # -----------------------------------------------------------------------
  run_config tp2 2 1 "" "$load_point" "$active_bs" "$num_requests" "$point_tag"
  run_config pp2_uniform 1 2 "18,18" "$load_point" "$active_bs" "$num_requests" "$point_tag"
  run_config pp2_auto 1 2 "23,13" "$load_point" "$active_bs" "$num_requests" "$point_tag"
done

if ((REPEATS == 1)) && [[ -f $SCRIPT_DIR/summarize_spectre_matrix.py ]]; then
  python "$SCRIPT_DIR/summarize_spectre_matrix.py" "$OUTPUT_ROOT"
fi
echo "All benchmark points complete: $OUTPUT_ROOT"
