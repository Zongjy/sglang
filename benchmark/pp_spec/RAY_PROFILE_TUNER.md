# RayEngine PP Boundary Tuner

This tuner deliberately reuses SGLang and Ray for runtime concerns:

```text
Ray placement group
  -> SGLang RayEngine
    -> SGLang offline workload
      -> SGLang Torch profiler traces
        -> offline PP boundary model
```

The tuner does not SSH to worker nodes, reproduce SGLang rank assignment, or
manage remote PID files. A multi-node profile directory must be mounted at the
same path on every Ray node.

## Runtime setup

Start or connect to a Ray cluster before profiling. The profile command uses
SGLang's existing `offline_throughput` Ray backend. For `nnodes > 1`, that
backend reserves one placement-group bundle per node with `STRICT_SPREAD`.

The formal `--pp-layer-partition` ServerArg is serialized with every
`SchedulerActor`. `SGLANG_PP_LAYER_PARTITION` remains a legacy fallback.

Example:

```bash
python benchmark/pp_spec/adaptive_pp_tuner.py profile \
  --output-dir /shared/profiles/qwen35-bs32 \
  --model-path Qwen/Qwen3.5-27B \
  --draft-model-path z-lab/Qwen3.5-27B-DFlash \
  --nnodes 2 --tp-size 1 --pp-size 4 \
  --current-partition 16,16,16,16 \
  --batch-size 32 --execution-bucket 8 \
  --profile-steps 32 --offline \
  --server-args --disable-radix-cache --enable-linear-replayssm-spec
```

`--execution-bucket` labels the CUDA-graph/microbatch bucket represented by a
run. Its default is `ceil(batch-size / pp-size)`, which assumes the default
`pp_async_batch_depth=0`; pass it explicitly for a different PP schedule or
when graph padding selects a larger bucket.

Capture more than one execution bucket by running `profile` into distinct
directories, then analyze them together:

```bash
python benchmark/pp_spec/adaptive_pp_tuner.py analyze \
  --profile-dir /shared/profiles/qwen35-bs16 \
  --profile-dir /shared/profiles/qwen35-bs32 \
  --target-batch-size 8 \
  --boundary-radius 8 \
  --min-layers 7 \
  --max-layers-per-rank 24,24,24,20
```

If the baseline gives every stage the same GDN/full ratio, add one nearby
partition at the same execution bucket. The analyzer uses it only to identify
typed layer costs; the first partition remains the optimization baseline:

```bash
python benchmark/pp_spec/adaptive_pp_tuner.py profile \
  --output-dir /shared/profiles/qwen35-bs32-l17 \
  --nnodes 2 --tp-size 1 --pp-size 4 \
  --current-partition 17,17,17,13 --batch-size 32

python benchmark/pp_spec/adaptive_pp_tuner.py analyze \
  --profile-dir /shared/profiles/qwen35-bs32 \
  --profile-dir /shared/profiles/qwen35-bs32-l17
```

## Cost model

For a fixed TP/PP device topology and execution bucket `b`, the profiler
produces a distribution of post-overlap GPU busy times for each PP stage:

```text
S_r(b) = measured service time of PP rank r
```

The parser uses consecutive `run_batch` ranges as iteration windows and takes
the union of GPU kernel/memcpy intervals in each window. Union time is used
instead of summing kernels so work on concurrent CUDA streams is not counted
twice. For TP stages, the slowest TP shard is the conservative stage sample.

DFlash adds three profiler ranges:

```text
sglang.dflash.target_prefill_forward
sglang.dflash.target_verify_forward
sglang.dflash.draft_model_forward
```

When Kineto correlation IDs connect GPU work to these ranges, target and draft
times are fitted separately. Otherwise the model uses total service time and
excludes the last rank from the layer-cost fit because that rank owns the
draft model.

For Qwen3.5's hybrid layout, target cost is represented as:

```text
T_r(b) = n_gdn(r) * c_gdn(b) + n_full(r) * c_full(b)
S_r(b) = T_r(b) + F_role(r, b)
```

`F_role` is the measured fixed residual. The last-stage residual includes
DFlash proposal, KV materialization, sampling, bookkeeping, and any GPU work
not attributed to target layers. Typed costs are fitted from middle stages when
`PP > 2`, avoiding embedding and output-head endpoint overhead. If the profiled
stages do not contain enough independent GDN/full mixtures for a two-variable
fit, one average per-layer cost is used and reported as a warning.

For a candidate partition `p`, the current optimizer predicts:

```text
S_hat_r(p, b) = typed_layer_cost(range_r(p), b) + F_role(r, b) + C_r(b)
J(p, b)       = max_r S_hat_r(p, b)
```

The objective is the bottleneck stage, not the sum of stage times. This matches
steady-state inference throughput. Communication `C_r` can be supplied as one
legacy scalar with `--t-comm-ms`, or as `P-1` boundary costs / `P` rank costs
with `--stage-comm-ms`. With a fixed SGLang rank-to-device mapping, changing
layer boundaries does not change which physical PP link each rank uses, but a
cross-node boundary can be substantially slower than an intra-node boundary.

The deployable family remains intentionally small:

```text
(l, ..., l, L - (P - 1) * l)
```

This matches DFlash's fixed last-stage draft placement. `--boundary-radius`
restricts `l` to a neighborhood of the profiled baseline. Predictions outside
that neighborhood should be treated as extrapolation and profiled separately.

## HexGen insights

[HexGen](https://arxiv.org/pdf/2311.11514) models heterogeneous inference as
compute plus TP/PP communication under hard per-device memory constraints. It
then uses dynamic programming for a fixed pipeline and an evolutionary search
for global device layout.

Only part of that design is needed here:

- Keep computation, communication, and memory as separate quantities.
- Treat memory as a hard feasibility constraint, never as a weighted score.
- Use contiguous layer ranges and a bottleneck-aware partition search.
- Drop the evolutionary device search: Ray has already fixed the homogeneous
  device placement and SGLang currently uses one TP degree across PP stages.
- Drop asymmetric per-stage TP: supporting it would require a different
  SGLang process-group and activation-broadcast design.

The current implementation exposes `--max-layers-per-rank` as a conservative
hard constraint. Without it, the report explicitly marks memory as
uncalibrated. A later extension should consume per-rank startup memory facts
from RayEngine rather than infer capacity from a single global GPU budget.

## Tessera insights

[Tessera](https://www.usenix.org/system/files/osdi26-hu-weifang.pdf) shows that
serial layer costs can select the wrong partition when communication overlap
depends on the exact heterogeneous layer combination. Its key transferable
idea is `profile -> bounded candidates -> select using measured post-overlap
cost`, with profile caching by chunk signature and device topology.

Applied to SGLang inference:

- Profile the real PP event loop with CUDA graphs and DFlash enabled; do not
  build the primary model from isolated eager layer timings.
- Use post-overlap stage service distributions, not serial sums.
- Search a bounded boundary neighborhood and validate the top one or two
  candidates on hardware.
- Feed every profiled neighbor back through the model and record predicted vs.
  measured bottleneck error; require another nearby profile when error exceeds
  10%.
- Cache profiles by model revision, GPU/interconnect class, TP/PP sizes,
  execution bucket, block size, and stage layer signature.
- Use an indifference band based on observed variance so noise does not cause
  repartition churn.

Tessera's dynamic bubble optimizer does not transfer directly. It fills
training bubbles with movable backward/weight-gradient work; autoregressive
SGLang inference has no equivalent pool of correctness-independent movable
tasks. DFlash acceptance variation is analogous to runtime stochasticity, but
the appropriate response here is robust quantiles and candidate validation,
not task injection.

## Known limitations

- Standard Torch traces are heavier than the old lightweight PPM stream.
- CUDA/Kineto correlation may fail to attribute target and draft kernels; the
  analyzer reports this fallback explicitly.
- The current boundary family is DFlash-specific, not a general linear
  partition solver.
- Memory is enforced only when `--max-layers-per-rank` is supplied.
- Trace directories must currently be on shared storage.
- Profiling one baseline cannot learn arbitrary pairwise layer-composition
  interactions. If validation shows systematic error, profile a small boundary
  neighborhood and cache composition-specific correction terms rather than
  adding a full runtime scheduler.
