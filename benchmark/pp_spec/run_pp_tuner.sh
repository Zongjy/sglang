#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
DRAFT_MODEL=${DRAFT_MODEL:-z-lab/Qwen3.5-9B-DFlash}
PP_SIZE=${PP_SIZE:-2}
TP_SIZE=${TP_SIZE:-1}
NNODES=${NNODES:-1}
BATCH_SIZE=${BATCH_SIZE:-32}
INPUT_TOKENS=${INPUT_TOKENS:-256}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-128}
PROFILE_STEPS=${PROFILE_STEPS:-32}
BLOCK_SIZE=${BLOCK_SIZE:-16}
PAGE_SIZE=${PAGE_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
MAMBA_SSM_DTYPE=${MAMBA_SSM_DTYPE:-float32}
MAMBA_FULL_MEMORY_RATIO=${MAMBA_FULL_MEMORY_RATIO:-0.9}
ENABLE_REPLAY_SSM=${ENABLE_REPLAY_SSM:-1}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRIPT_DIR/results/${MODEL//\//_}_profile_$(date -u +%Y%m%d_%H%M%S)}

ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
DRAFT_ATTENTION_BACKEND=${DRAFT_ATTENTION_BACKEND:-flashinfer}
DTYPE=${DTYPE:-bfloat16}

if [[ -z ${CURRENT_PARTITION:-} ]]; then
  echo "CURRENT_PARTITION is required (for example: CURRENT_PARTITION=16,16)" >&2
  exit 2
fi
if [[ $ENABLE_REPLAY_SSM != 0 && $ENABLE_REPLAY_SSM != 1 ]]; then
  echo "ENABLE_REPLAY_SSM must be 0 or 1" >&2
  exit 2
fi

profile_args=(
  profile
  --output-dir "$OUTPUT_DIR"
  --model-path "$MODEL"
  --draft-model-path "$DRAFT_MODEL"
  --pp-size "$PP_SIZE"
  --tp-size "$TP_SIZE"
  --nnodes "$NNODES"
  --current-partition "$CURRENT_PARTITION"
  --batch-size "$BATCH_SIZE"
  --input-tokens "$INPUT_TOKENS"
  --output-tokens "$OUTPUT_TOKENS"
  --profile-steps "$PROFILE_STEPS"
  --block-size "$BLOCK_SIZE"
  --page-size "$PAGE_SIZE"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --mamba-ssm-dtype "$MAMBA_SSM_DTYPE"
  --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO"
)

if [[ -n ${EXECUTION_BUCKET:-} ]]; then
  profile_args+=(--execution-bucket "$EXECUTION_BUCKET")
fi
if [[ -n ${MAX_RUNNING_REQUESTS:-} ]]; then
  profile_args+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
fi
if [[ ${OFFLINE:-0} == 1 ]]; then
  profile_args+=(--offline)
fi

echo "RayEngine PP profile output: $OUTPUT_DIR"
cd "$REPO_ROOT"
server_args=(
  --dtype "$DTYPE"
  --attention-backend "$ATTENTION_BACKEND"
  --speculative-draft-attention-backend "$DRAFT_ATTENTION_BACKEND"
  --disable-radix-cache
)
if [[ $ENABLE_REPLAY_SSM == 1 ]]; then
  server_args+=(--linear-attn-backend triton --enable-linear-replayssm-spec)
fi
exec python "$SCRIPT_DIR/adaptive_pp_tuner.py" \
  "${profile_args[@]}" \
  "$@" \
  --server-args \
  "${server_args[@]}"
