#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

MODEL=${MODEL:-Qwen/Qwen3-8B}
DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}
DATASET=${DATASET:-$SCRIPT_DIR/data/sharegpt.json}
GPUS=${GPUS:-0,1}
PP_SIZE=${PP_SIZE:-2}
TP_SIZE=${TP_SIZE:-1}
BLOCK_SIZE=${BLOCK_SIZE:-16}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-32}
CONCURRENCY=${CONCURRENCY:-32}
PROMPT_TOKENS=${PROMPT_TOKENS:-256}
DECODE_TOKENS_PER_REQUEST=${DECODE_TOKENS_PER_REQUEST:-256}
PPM_COLLECT_S=${PPM_COLLECT_S:-30}
MEMORY_RESERVE_GIB=${MEMORY_RESERVE_GIB:-2.0}
REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-1800}
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-600}
COMM_BENCHMARK_TOKENS=${COMM_BENCHMARK_TOKENS:-64,128,256,512,1024}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/results/${MODEL}_pp_partition_tune_$(date -u +%Y%m%d_%H%M%S)}

ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
DRAFT_ATTENTION_BACKEND=${DRAFT_ATTENTION_BACKEND:-flashinfer}
DTYPE=${DTYPE:-bfloat16}

[[ -f $DATASET ]] || {
  echo "ShareGPT dataset not found: $DATASET" >&2
  exit 1
}

tuner_args=(
  --pp-size "$PP_SIZE"
  --tp-size "$TP_SIZE"
  --model-path "$MODEL"
  --draft-model-path "$DRAFT_MODEL"
  --block-size "$BLOCK_SIZE"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
  --concurrency "$CONCURRENCY"
  --prompt-tokens "$PROMPT_TOKENS"
  --decode-tokens-per-request "$DECODE_TOKENS_PER_REQUEST"
  --dataset "$DATASET"
  --visible-devices "$GPUS"
  --ppm-collect-s "$PPM_COLLECT_S"
  --memory-reserve-gib "$MEMORY_RESERVE_GIB"
  --request-timeout-s "$REQUEST_TIMEOUT_S"
  --server-startup-timeout-s "$STARTUP_TIMEOUT_S"
  --comm-benchmark-tokens "$COMM_BENCHMARK_TOKENS"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n ${CURRENT_PARTITION:-} ]]; then
  tuner_args+=(--current-partition "$CURRENT_PARTITION")
fi
if [[ -n ${FIXED_ACTIVE_REQUESTS:-} ]]; then
  tuner_args+=(--fixed-active-requests "$FIXED_ACTIVE_REQUESTS")
fi
if [[ -n ${CAPTURE_BUCKETS:-} ]]; then
  tuner_args+=(--capture-buckets "$CAPTURE_BUCKETS")
fi
if [[ -n ${MAMBA_SSM_DTYPE:-} ]]; then
  tuner_args+=(--mamba-ssm-dtype "$MAMBA_SSM_DTYPE")
fi
if [[ -n ${MAMBA_FULL_MEMORY_RATIO:-} ]]; then
  tuner_args+=(--mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO")
fi
if [[ -n ${T_COMM_MS:-} ]]; then
  tuner_args+=(--t-comm-ms "$T_COMM_MS")
fi

echo "PP tuner output: $OUTPUT_DIR"
cd "$REPO_ROOT"
exec python "$SCRIPT_DIR/adaptive_pp_tuner.py" \
  "${tuner_args[@]}" \
  "$@" \
  --server-args \
  --dtype "$DTYPE" \
  --attention-backend "$ATTENTION_BACKEND" \
  --speculative-draft-attention-backend "$DRAFT_ATTENTION_BACKEND" \
  --disable-radix-cache
