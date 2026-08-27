# RayEngine PP Boundary Tuner

The tuner has one workflow:

```text
one baseline RayEngine profile
  -> per-rank Kineto traces
  -> remove PP Send/Recv GPU intervals
  -> fit target-layer and fixed stage costs
  -> enumerate (l, ..., l, residual)
  -> select the lowest predicted bottleneck
```

## Profile

Start or connect to the Ray cluster, then capture one baseline partition:

```bash
python benchmark/pp_spec/adaptive_pp_tuner.py profile \
  --output-dir /shared/profiles/qwen35-bs32 \
  --model-path Qwen/Qwen3.5-27B \
  --draft-model-path z-lab/Qwen3.5-27B-DFlash \
  --nnodes 2 --tp-size 1 --pp-size 4 \
  --current-partition 16,16,16,16 \
  --batch-size 128 --execution-bucket 32 \
  --block-size 16 --mem-fraction-static 0.75 \
  --mamba-ssm-dtype float32 --mamba-full-memory-ratio 0.9 \
  --profile-steps 32 --offline \
  --server-args --disable-radix-cache --linear-attn-backend triton \
  --enable-linear-replayssm-spec
```

For multi-node profiling, the output directory must be visible at the same
path on every Ray node.

## Analyze

```bash
python benchmark/pp_spec/adaptive_pp_tuner.py analyze \
  --profile-dir /shared/profiles/qwen35-bs32 \
  --boundary-radius 8 \
  --min-layers 7
```

The analyzer writes `analysis.json`, `analysis.txt`, and `recommended.args`.

## Cost Model

For each steady-state verify window, the trace parser computes:

```text
intrinsic GPU work = union(all GPU events except pp:device Send/Recv)
target GPU work    = GPU events correlated with target_verify_forward
```

The trace set must contain exactly one file for every `(PP rank, TP rank)` and
must include Kineto process-group metadata. The parser does not use kernel-name
or CPU-time inference.

The analytical stage model is:

```text
stage_r(partition)
  = typed_target_layer_cost(range_r)
  + baseline_role_fixed_cost_r
  + optional_stage_comm_floor_r

objective(partition) = max_r stage_r(partition)
```

The optimizer enumerates `(l, ..., l, L - (P - 1) * l)` and returns the
candidate with the smallest predicted objective.
