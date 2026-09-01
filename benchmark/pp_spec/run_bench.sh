#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# Shared config. ReplaySSM is always on (hybrid linear-attention models).
MODEL_REVISION=main
DRAFT_MODEL_REVISION=main
DATASET=$SCRIPT_DIR/data/sharegpt.json
GPUS=0,1
NUM_GPUS=2
PORT=31000
RAGGED_VERIFY_MODE=static
MAX_TOKENS=1024
PROMPT_MAX_TOKENS=2000
DFLASH_BLOCK_SIZE=16
STARTUP_TIMEOUT=600
REQUEST_TIMEOUT_S=1800
COOLDOWN_S=10
BASE_URL="http://127.0.0.1:$PORT"
SERVER_PID=
FAILED_RUNS=()

# ---------------------------------------------------------------------------
# Part 1: helpers
# ---------------------------------------------------------------------------

cleanup() {
  local pid=${SERVER_PID:-}
  SERVER_PID=
  [[ -n $pid ]] || return 0

  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in {1..60}; do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
      sleep 5
    fi
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
}

on_signal() {
  local status=$1
  trap - INT TERM
  exit "$status"
}
trap cleanup EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

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

record_failure() {
  local target=$1
  local phase=$2
  local status=$3

  FAILED_RUNS+=("$target: $phase (exit $status)")
  printf '%s\n' "$phase (exit $status)" >"$target/FAILED"
  echo "[$target] $phase failed with exit code $status; continuing." >&2
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
  local status=0
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
  cuda_graph_bs=$((cuda_graph_bs < 64 ? cuda_graph_bs : 64))
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
    --disable-prefill-cuda-graph
    --cuda-graph-max-bs-decode "$cuda_graph_bs"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --page-size 1
    --random-seed 1
    --disable-radix-cache
    --trust-remote-code
    --linear-attn-backend triton
    --mamba-ssm-dtype float32
    --mamba-full-memory-ratio 0.9
    --enable-linear-replayssm-spec
    --host 127.0.0.1
    --port "$PORT"
  )
  if [[ -n $partition ]]; then
    server_args+=(--pp-layer-partition "$partition")
  fi
  if ((pp_size > 1)); then
    server_args+=(--disable-overlap-schedule)
  fi
  server_args+=("${extra_server_args[@]}")

  echo "[$name][$load_point] starting: tp=$tp_size pp=$pp_size dp=$dp_size max_running=$max_running_requests cuda_graph_bs=$cuda_graph_bs mem_fraction=$MEM_FRACTION_STATIC"
  CUDA_VISIBLE_DEVICES="$GPUS" \
    SGLANG_RAGGED_VERIFY_MODE="$RAGGED_VERIFY_MODE" \
    SGL_FORCE_SHUTDOWN=1 \
    setsid sglang "${server_args[@]}" >"$output_dir/server.log" 2>&1 &
  SERVER_PID=$!
  if wait_for_server "$output_dir/server.log"; then
    :
  else
    status=$?
    cleanup
    record_failure "$output_dir" "server startup" "$status"
    return 0
  fi

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
    --output-dir "$output_dir" || status=$?

  cleanup
  if ((status != 0)); then
    record_failure "$output_dir" "benchmark" "$status"
    return 0
  fi
  echo "[$name][$load_point] complete"
}

summarize_run() {
  local run_dir=$1
  local status=0

  python "$SCRIPT_DIR/summarize_spectre.py" "$run_dir" || status=$?
  if ((status != 0)); then
    record_failure "$run_dir" "summary" "$status"
    return 0
  fi

  python "$SCRIPT_DIR/plot_performance.py" "$run_dir/summary.csv" || status=$?
  if ((status != 0)); then
    record_failure "$run_dir" "plotting" "$status"
  fi
}

# ---------------------------------------------------------------------------
# Part 2: the runs
# ---------------------------------------------------------------------------

cd "$REPO_ROOT"

# # ===== Qwen3.5-9B (32 layers; uniform 16,16 / auto 20,12) =====
# MODEL=Qwen/Qwen3.5-9B
# DRAFT_MODEL=z-lab/Qwen3.5-9B-DFlash
# MEM_FRACTION_STATIC=0.75
# OUTPUT_ROOT=$SCRIPT_DIR/results/Qwen_Qwen3.5-9B_$(date -u +%Y%m%d_%H%M%S)
# mkdir -p "$OUTPUT_ROOT"
# echo "Results: $OUTPUT_ROOT"

# # run_config name <tp size> <pp_size> <dp_size> <load_point:并发:QPS:请求数> <active_bs:全局并发> <num_requests:总请求数> <point_tag:结果目录后缀c{C}_qps{Q}_n{N}>
# run_config tp2 2 1 1 "" 8:2:32 8 32 c8_qps2_n32
# run_config dp2 1 1 2 "" 8:2:32 8 32 c8_qps2_n32
# run_config pp2_uniform 1 2 1 16,16 8:2:32 8 32 c8_qps2_n32
# run_config pp2_auto 1 2 1 20,12 8:2:32 8 32 c8_qps2_n32

# run_config tp2 2 1 1 "" 16:4:64 16 64 c16_qps4_n64
# run_config dp2 1 1 2 "" 16:4:64 16 64 c16_qps4_n64
# run_config pp2_uniform 1 2 1 16,16 16:4:64 16 64 c16_qps4_n64
# run_config pp2_auto 1 2 1 20,12 16:4:64 16 64 c16_qps4_n64

# run_config tp2 2 1 1 "" 32:8:128 32 128 c32_qps8_n128
# run_config dp2 1 1 2 "" 32:8:128 32 128 c32_qps8_n128
# run_config pp2_uniform 1 2 1 16,16 32:8:128 32 128 c32_qps8_n128
# run_config pp2_auto 1 2 1 20,12 32:8:128 32 128 c32_qps8_n128

# run_config tp2 2 1 1 "" 64:16:256 64 256 c64_qps16_n256
# run_config dp2 1 1 2 "" 64:16:256 64 256 c64_qps16_n256
# run_config pp2_uniform 1 2 1 16,16 64:16:256 64 256 c64_qps16_n256
# run_config pp2_auto 1 2 1 20,12 64:16:256 64 256 c64_qps16_n256

# run_config tp2 2 1 1 "" 96:24:384 96 384 c96_qps24_n384
# run_config dp2 1 1 2 "" 96:24:384 96 384 c96_qps24_n384
# run_config pp2_uniform 1 2 1 16,16 96:24:384 96 384 c96_qps24_n384
# run_config pp2_auto 1 2 1 20,12 96:24:384 96 384 c96_qps24_n384

# run_config tp2 2 1 1 "" 128:32:512 128 512 c128_qps32_n512
# run_config dp2 1 1 2 "" 128:32:512 128 512 c128_qps32_n512
# run_config pp2_uniform 1 2 1 16,16 128:32:512 128 512 c128_qps32_n512
# run_config pp2_auto 1 2 1 20,12 128:32:512 128 512 c128_qps32_n512

# summarize_run "$OUTPUT_ROOT"

# ===== Qwen3.5-27B-FP8 (64 layers; uniform 32,32 / auto 38,26) =====
MODEL=Qwen/Qwen3.5-27B-FP8
DRAFT_MODEL=z-lab/Qwen3.5-27B-DFlash
MEM_FRACTION_STATIC=0.8
OUTPUT_ROOT=$SCRIPT_DIR/results/Qwen_Qwen3.5-27B-FP8_$(date -u +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_ROOT"
echo "Results: $OUTPUT_ROOT"

# run_config <name:结果子目录> <tp> <pp> <dp> <partition:PP分层,空=非PP> <load_point:并发:QPS:请求数> <active_bs:全局并发> <num_requests:总请求数> <point_tag:结果目录后缀c{C}_qps{Q}_n{N}>
run_config tp2 2 1 1 "" 8:2:32 8 32 c8_qps2_n32
run_config dp2 1 1 2 "" 8:2:32 8 32 c8_qps2_n32
run_config pp2_uniform 1 2 1 32,32 8:2:32 8 32 c8_qps2_n32
run_config pp2_auto 1 2 1 38,26 8:2:32 8 32 c8_qps2_n32

run_config tp2 2 1 1 "" 16:4:64 16 64 c16_qps4_n64
run_config dp2 1 1 2 "" 16:4:64 16 64 c16_qps4_n64
run_config pp2_uniform 1 2 1 32,32 16:4:64 16 64 c16_qps4_n64
run_config pp2_auto 1 2 1 38,26 16:4:64 16 64 c16_qps4_n64

run_config tp2 2 1 1 "" 32:8:128 32 128 c32_qps8_n128
run_config dp2 1 1 2 "" 32:8:128 32 128 c32_qps8_n128
run_config pp2_uniform 1 2 1 32,32 32:8:128 32 128 c32_qps8_n128
run_config pp2_auto 1 2 1 38,26 32:8:128 32 128 c32_qps8_n128

run_config tp2 2 1 1 "" 64:16:256 64 256 c64_qps16_n256
run_config dp2 1 1 2 "" 64:16:256 64 256 c64_qps16_n256
run_config pp2_uniform 1 2 1 32,32 64:16:256 64 256 c64_qps16_n256
run_config pp2_auto 1 2 1 38,26 64:16:256 64 256 c64_qps16_n256

run_config tp2 2 1 1 "" 96:24:384 96 384 c96_qps24_n384
run_config dp2 1 1 2 "" 96:24:384 96 384 c96_qps24_n384
run_config pp2_uniform 1 2 1 32,32 96:24:384 96 384 c96_qps24_n384
run_config pp2_auto 1 2 1 38,26 96:24:384 96 384 c96_qps24_n384

run_config tp2 2 1 1 "" 128:32:512 128 512 c128_qps32_n512
run_config dp2 1 1 2 "" 128:32:512 128 512 c128_qps32_n512
run_config pp2_uniform 1 2 1 32,32 128:32:512 128 512 c128_qps32_n512
run_config pp2_auto 1 2 1 38,26 128:32:512 128 512 c128_qps32_n512

summarize_run "$OUTPUT_ROOT"

if ((${#FAILED_RUNS[@]} > 0)); then
  printf 'Benchmark sweep completed with %d failure(s):\n' "${#FAILED_RUNS[@]}" >&2
  printf '  - %s\n' "${FAILED_RUNS[@]}" >&2
  exit 1
fi

echo "All benchmark points complete."
