#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

if [[ -z ${PYTHON:-} ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON=python
  fi
fi
MODEL=${MODEL:-Qwen/Qwen3-8B}
DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3-8B-DFlash-b16}
PP_SIZE=${PP_SIZE:-2}
TP_SIZE=${TP_SIZE:-1}
NNODES=${NNODES:-1}
BATCH_SIZES=${BATCH_SIZES:-${BATCH_SIZE:-"32 64 128"}}
INPUT_TOKENS=${INPUT_TOKENS:-2000}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-256}
PROFILE_STEPS=${PROFILE_STEPS:-128}
BLOCK_SIZE=${BLOCK_SIZE:-16}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.7}
PAGE_SIZE=${PAGE_SIZE:-1}
MAMBA_SSM_DTYPE=${MAMBA_SSM_DTYPE:-float32}
MAMBA_FULL_MEMORY_RATIO=${MAMBA_FULL_MEMORY_RATIO:-0.9}
ENABLE_REPLAY_SSM=${ENABLE_REPLAY_SSM:-1}
OFFLINE=${OFFLINE:-0}
DTYPE=${DTYPE:-bfloat16}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-triton}
DRAFT_ATTENTION_BACKEND=${DRAFT_ATTENTION_BACKEND:-flashinfer}
MIN_LAYERS=${MIN_LAYERS:-8}
K_BEST=${K_BEST:-30}
DRY_RUN=${DRY_RUN:-0}
# Measure PP edge latency with the runtime's payload-shaped ping-pong
# benchmark.  D-Cut consumes the resulting alpha/beta model in memory.
export SGLANG_PP_COMM_BENCHMARK=${SGLANG_PP_COMM_BENCHMARK:-1}
export SGLANG_PP_COMM_BENCHMARK_TOKENS=${SGLANG_PP_COMM_BENCHMARK_TOKENS:-64,256,1024,4096}

MODEL_TAG=${MODEL//\//_}
RESULTS_DIR=${RESULTS_DIR:-${OUTPUT_DIR:-$SCRIPT_DIR/results/${MODEL_TAG}_multibatch_$(date -u +%Y%m%d_%H%M%S)}}

read -r -a batch_sizes <<< "$BATCH_SIZES"
if [[ ${#batch_sizes[@]} -eq 0 ]]; then
  echo "BATCH_SIZES must contain at least one batch size" >&2
  exit 2
fi

args=(
  --results-dir "$RESULTS_DIR"
  --batch-sizes "${batch_sizes[@]}"
  --model-path "$MODEL"
  --draft-model-path "$DRAFT_MODEL"
  --tp-size "$TP_SIZE"
  --pp-size "$PP_SIZE"
  --nnodes "$NNODES"
  --input-tokens "$INPUT_TOKENS"
  --output-tokens "$OUTPUT_TOKENS"
  --profile-steps "$PROFILE_STEPS"
  --block-size "$BLOCK_SIZE"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --page-size "$PAGE_SIZE"
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE"
  --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO"
  --dtype "$DTYPE"
  --attention-backend "$ATTENTION_BACKEND"
  --draft-attention-backend "$DRAFT_ATTENTION_BACKEND"
  --min-layers "$MIN_LAYERS"
  --k-best "$K_BEST"
)

if [[ -n ${BASELINE_PARTITION:-} ]]; then
  args+=(--baseline-partition "$BASELINE_PARTITION")
fi
if [[ ${DRY_RUN} == 1 ]]; then
  args+=(--dry-run)
fi
if [[ ${ENABLE_REPLAY_SSM} != 1 && ${ENABLE_REPLAY_SSM} != 0 ]]; then
  echo "ENABLE_REPLAY_SSM must be 0 or 1" >&2
  exit 2
fi
if [[ ${ENABLE_REPLAY_SSM} == 0 ]]; then
  args+=(--disable-replay-ssm)
fi
if [[ ${OFFLINE} == 1 ]]; then
  args+=(--offline)
fi

echo "PP multibatch tuner output: $RESULTS_DIR"
cd "$REPO_ROOT"
exec "$PYTHON" "$SCRIPT_DIR/run_multibatch_profile.py" "${args[@]}" "$@"
