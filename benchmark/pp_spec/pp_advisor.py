#!/usr/bin/env python3
"""PP partition advisor: continuously re-evaluate a running production server.

Each cycle the advisor collects a PPM snapshot from the live server's ZMQ
endpoints, refits the stage cost model, recalibrates the capacity model from
the server log, and re-solves the one-dimensional cycle-time objective.  Raw
Mamba/KV capacity refines only the latency-equivalent candidates.  When the
recommended partition beats the current one by more than --min-gain-pct AND
the gain is outside the noise band, it prints an actionable
``export SGLANG_PP_LAYER_PARTITION=...`` suggestion with the full reasoning;
otherwise it prints a one-line heartbeat.

Low-load periods are fine: when the collection window yields too few
work-conserving samples, the cycle is skipped (not an error).  ``--once``
runs a single evaluation, suitable for cron or smoke tests.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.5-9B-DFlash"


class AdvisorError(RuntimeError):
    pass


@dataclass
class CycleOutcome:
    action: str  # "recommend" | "heartbeat" | "skipped"
    detail: str
    current_partition: tuple[int, ...]
    recommended_partition: tuple[int, ...] | None = None
    gain_pct: float | None = None


class Advisor:
    """Holds cached endpoints / capacity context across cycles."""

    def __init__(self, args: argparse.Namespace) -> None:
        sys.path.insert(0, str(SCRIPT_DIR))
        self.args = args
        self.current_partition = tuple(
            int(item.strip()) for item in args.current_partition.split(",")
        )
        pp_size = args.pp_size or len(self.current_partition)
        if len(self.current_partition) != pp_size or any(
            count <= 0 for count in self.current_partition
        ):
            raise AdvisorError(
                f"--current-partition {args.current_partition!r} must contain "
                f"{pp_size} positive layer counts."
            )
        self.pp_size = pp_size
        self.num_layers = sum(self.current_partition)
        self._endpoints: list[str] | None = list(args.endpoint or []) or None
        self._capacity_ctx: dict[str, Any] | None = None
        self._capacity_attempted = False

    def endpoints(self) -> list[str]:
        if self._endpoints:
            return self._endpoints
        import ppm_consumer

        text = self._server_log_text()
        endpoints = ppm_consumer.parse_ppm_endpoints(text)
        if not endpoints:
            raise AdvisorError(
                "no 'PPM: ZMQ PUB bound on' lines in the server log; pass "
                "--endpoint explicitly"
            )
        self._endpoints = endpoints
        return endpoints

    def _server_log_text(self) -> str:
        if not self.args.server_log:
            return ""
        try:
            return Path(self.args.server_log).read_text(errors="replace")
        except OSError:
            return ""

    def capacity_ctx(self) -> dict[str, Any] | None:
        """Build once per process; None when the log/model is insufficient."""
        if self._capacity_attempted:
            return self._capacity_ctx
        self._capacity_attempted = True
        if not self.args.server_log:
            print(
                "[advisor] no --server-log; capacity model disabled, using "
                "the compute-only objective",
                flush=True,
            )
            return None
        if getattr(self.args, "tp_size", 1) != 1:
            print(
                "[advisor] capacity model disabled for TP>1: its static byte "
                "counts are not TP-shard aware",
                flush=True,
            )
            return None
        import capacity_model

        try:
            static = capacity_model.load_static_info(
                self.args.model_path,
                self.args.draft_model_path,
                state_dtype=getattr(self.args, "mamba_ssm_dtype", None),
            )
            calibration = capacity_model.calibrate(
                self._server_log_text(),
                static,
                self.current_partition,
                self.args.mem_fraction_static,
                self.args.mamba_full_memory_ratio,
                self.args.block_size,
                safety_gib=getattr(self.args, "memory_reserve_gib", 1.0),
                tokens_per_request=getattr(self.args, "tokens_per_request", 512),
                page_size=getattr(self.args, "page_size", 1),
            )
            if (
                calibration.pre_avail_gib <= 0
                and not any(value > 0 for value in calibration.baseline_post_avail_gib)
            ):
                print(
                    "[advisor] capacity model disabled: server log has no "
                    "usable memory calibration lines",
                    flush=True,
                )
                return None
        except capacity_model.CapacityModelError as exc:
            print(f"[advisor] capacity model disabled: {exc}", flush=True)
            return None

        target_requests = (
            getattr(self.args, "target_global_requests", None)
            or calibration.resolved_max_running
            or 1
        )
        tokens_per_request = getattr(self.args, "tokens_per_request", 512)

        def capacity_of(partition: Sequence[int]):
            return capacity_model.predict_capacity(
                partition,
                static,
                calibration,
                target_requests=target_requests,
                tokens_per_request=tokens_per_request,
                draft_tokens=self.args.block_size,
                safety_gib=getattr(self.args, "memory_reserve_gib", None),
            )

        self._capacity_ctx = {
            "static": static,
            "capacity_of": capacity_of,
            "target_requests": target_requests,
            "tokens_per_request": tokens_per_request,
        }
        return self._capacity_ctx

    def evaluate(
        self,
        collect_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> CycleOutcome:
        """One evaluation cycle.  ``collect_fn`` is injectable for tests."""
        import ppm_consumer
        import partition_optimizer
        import stage_model
        from model_layout import LayerLayout

        endpoints = self.endpoints()
        if collect_fn is None:
            capture_buckets = None
            if getattr(self.args, "capture_buckets", None):
                capture_buckets = tuple(
                    int(value.strip())
                    for value in self.args.capture_buckets.split(",")
                    if value.strip()
                )
            elif getattr(self.args, "cuda_graph_max_bs", None):
                capture_buckets = ppm_consumer.default_capture_buckets(
                    self.args.cuda_graph_max_bs, speculative=True
                )
            collect_fn = lambda eps: ppm_consumer.collect(  # noqa: E731
                eps,
                duration_s=self.args.collect_s,
                capture_buckets=capture_buckets,
            )
        snapshot = collect_fn(endpoints)

        layout_warning: str | None = None
        try:
            layout = LayerLayout.from_model_path(
                self.args.model_path, local_files_only=True
            )
        except Exception as exc:
            layout = LayerLayout.from_kinds(("full",) * self.num_layers)
            layout_warning = (
                "layer layout unavailable; latency model assumes all-full layers "
                f"({exc})"
            )
        model = stage_model.StageCostModel.fit(
            snapshot,
            self.current_partition,
            min_samples=self.args.min_samples,
            layout=layout,
            capture_buckets=snapshot.get("capture_buckets"),
        )
        if layout_warning:
            model.warnings.append(layout_warning)
        capacity_ctx = self.capacity_ctx()

        target_global = (
            getattr(self.args, "target_global_requests", None)
            or (self._capacity_ctx or {}).get("target_requests")
            or model.target_bucket().bucket * max(model.pp_loop_size, 1)
        )
        target_bs = max(1, math.ceil(target_global / max(model.pp_loop_size, 1)))
        estimate = model.estimate_for_bs(target_bs)
        measured_max_bucket = max(model.buckets)
        if target_bs > measured_max_bucket:
            model.warnings.append(
                f"target per-slot bs {target_bs} exceeds the largest measured "
                f"bucket {measured_max_bucket}; using that bucket without "
                "interpolation"
            )
        elif target_bs != estimate.bucket:
            model.warnings.append(
                f"target per-slot bs {target_bs} is evaluated at execution "
                f"bucket {estimate.bucket}"
            )
        work_tokens = estimate.bucket * self.args.block_size
        if self.args.t_comm_ms is not None:
            t_comm_ms = self.args.t_comm_ms
            t_comm_source = "manual --t-comm-ms"
        else:
            parsed = ppm_consumer.t_comm_from_log(
                self._server_log_text(), work_tokens
            )
            if parsed is None:
                t_comm_ms = 0.0
                t_comm_source = "unavailable; fell back to 0"
            else:
                t_comm_ms = parsed
                t_comm_source = "startup comm benchmark"

        try:
            result = partition_optimizer.optimize(
                model,
                estimate=estimate,
                t_comm_ms=t_comm_ms,
                k_best=self.args.k_best,
                capacity=capacity_ctx["capacity_of"] if capacity_ctx else None,
                target_bs=estimate.bucket,
                layout=layout,
            )
        except partition_optimizer.OptimizerError as exc:
            if capacity_ctx is None:
                raise
            raise AdvisorError(
                "no prefix-uniform partition satisfies the fixed memory "
                f"working point ({exc}); cycle skipped"
            ) from exc

        result.warnings.extend(model.warnings)

        best, current = result.best, result.current
        gain = (
            (current.cycle_time_ms - best.cycle_time_ms) / current.cycle_time_ms
            if current.cycle_time_ms > 0
            else 0.0
        )
        metric = (
            f"current cycle {current.cycle_time_ms:.2f} ms; best "
            f"{','.join(map(str, best.partition))}: "
            f"{best.cycle_time_ms:.2f} ms"
        )
        if best.memory_capacity is not None and current.memory_capacity is not None:
            metric += (
                f"; raw memory capacity {current.memory_capacity} -> "
                f"{best.memory_capacity}"
            )

        current_text = ",".join(map(str, self.current_partition))
        best_text = ",".join(map(str, best.partition))
        latency_band = {item.partition for item in result.indifference_set}
        capacity_refinement = (
            current.partition in latency_band
            and best.partition in latency_band
            and current.memory_capacity is not None
            and best.memory_capacity is not None
            and best.memory_capacity > current.memory_capacity
        )
        current_memory_infeasible = current.target_feasible is False
        if (
            not result.keep_current
            and best.partition != self.current_partition
            and (
                gain * 100.0 > self.args.min_gain_pct
                or capacity_refinement
                or current_memory_infeasible
            )
        ):
            if capacity_refinement:
                reason = (
                    "both partitions are inside the latency-equivalent range; "
                    "the candidate has more raw memory capacity"
                )
            elif current_memory_infeasible:
                reason = "the current partition fails memory feasibility"
            else:
                reason = (
                    f"predicted latency gain {gain:.1%} > --min-gain-pct "
                    f"{self.args.min_gain_pct}% and outside the "
                    f"{result.noise_sigma:g}-sigma noise band"
                )
            detail = (
                f"[advisor] RECOMMEND repartition {current_text} -> {best_text}\n"
                f"  {metric}\n"
                f"  {reason} "
                f"(t_comm={t_comm_ms:.4f} ms, {t_comm_source})\n"
                f"  apply: export SGLANG_PP_LAYER_PARTITION={best_text}"
            )
            return CycleOutcome(
                "recommend", detail, self.current_partition, best.partition, gain * 100
            )
        detail = (
            f"[advisor] heartbeat: current {current_text} still optimal "
            f"(best alternative {best_text} at {gain:+.1%} predicted, "
            f"threshold {self.args.min_gain_pct}%; {metric})"
        )
        return CycleOutcome(
            "heartbeat", detail, self.current_partition, best.partition, gain * 100
        )

    def run(self) -> None:
        while True:
            try:
                outcome = self.evaluate()
                print(outcome.detail, flush=True)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # low-load windows, transient zmq, ...
                print(
                    f"[advisor] cycle skipped: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            if self.args.once:
                return
            time.sleep(self.args.interval_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-log",
        help=(
            "production server log: auto-discovers PPM endpoints, the comm "
            "benchmark fit, and the capacity calibration facts"
        ),
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        help="explicit PPM PUB endpoint (repeatable); overrides --server-log discovery",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--draft-model-path", default=DEFAULT_DRAFT_MODEL)
    parser.add_argument(
        "--current-partition",
        required=True,
        help="live partition, e.g. '37,27' (required)",
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        help="default: inferred from --current-partition length",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=300.0)
    parser.add_argument("--collect-s", type=float, default=60.0)
    parser.add_argument("--min-gain-pct", type=float, default=8.0)
    parser.add_argument("--t-comm-ms", type=float)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--mamba-ssm-dtype", default="bfloat16")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=32)
    parser.add_argument("--capture-buckets")
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument(
        "--mamba-full-memory-ratio",
        type=float,
        default=2.0,
        help="fixed runtime Mamba/KV memory ratio used by the capacity model",
    )
    parser.add_argument("--memory-reserve-gib", type=float, default=1.0)
    parser.add_argument(
        "--target-global-requests",
        type=int,
        help="fixed global request working point for memory feasibility",
    )
    parser.add_argument(
        "--tokens-per-request",
        type=int,
        default=512,
        help="fixed KV tokens per request used by the memory filter",
    )
    parser.add_argument("--k-best", type=int, default=20)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="minimum samples per (rank, bucket) cell for the fit",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="evaluate a single cycle and exit (cron / smoke test)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        Advisor(args).run()
    except (AdvisorError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("[advisor] stopped", file=sys.stderr)
        else:
            print(f"[advisor] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
