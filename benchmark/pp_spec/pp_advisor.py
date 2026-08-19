#!/usr/bin/env python3
"""PP partition advisor: continuously re-evaluate a running production server.

Each cycle the advisor collects a PPM snapshot from the live server's ZMQ
endpoints, refits the stage cost model, recalibrates the capacity model from
the server log, and re-solves the unified objective
``throughput(p) = BS_max(p) x accept_len / cadence(p, BS_max(p))``.  When the
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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.6-27B-DFlash"


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
        import capacity_model

        try:
            static = capacity_model.load_static_info(
                self.args.model_path, self.args.draft_model_path
            )
            calibration = capacity_model.calibrate(
                self._server_log_text(),
                static,
                self.current_partition,
                self.args.mem_fraction_static,
                self.args.mamba_full_memory_ratio,
                self.args.block_size,
                self.args.cuda_graph_max_bs,
            )
        except capacity_model.CapacityModelError as exc:
            print(f"[advisor] capacity model disabled: {exc}", flush=True)
            return None

        def capacity_of(partition: Sequence[int]):
            return capacity_model.predict_capacity(
                partition,
                static,
                calibration,
                self.args.mamba_full_memory_ratio,
                self.args.block_size,
            )

        self._capacity_ctx = {"static": static, "capacity_of": capacity_of}
        return self._capacity_ctx

    def evaluate(
        self,
        collect_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> CycleOutcome:
        """One evaluation cycle.  ``collect_fn`` is injectable for tests."""
        import ppm_consumer
        import partition_optimizer
        import stage_model

        endpoints = self.endpoints()
        if collect_fn is None:
            collect_fn = lambda eps: ppm_consumer.collect(  # noqa: E731
                eps, duration_s=self.args.collect_s
            )
        snapshot = collect_fn(endpoints)

        model = stage_model.StageCostModel.fit(
            snapshot, self.current_partition, min_samples=self.args.min_samples
        )
        capacity_ctx = self.capacity_ctx()

        work_tokens = model.target_bucket().bs_mean * self.args.block_size
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

        result = partition_optimizer.optimize(
            model,
            t_comm_ms=t_comm_ms,
            k_best=self.args.k_best,
            capacity=capacity_ctx["capacity_of"] if capacity_ctx else None,
        )

        best, current = result.best, result.current
        if best.throughput_tok_s is not None and current.throughput_tok_s:
            gain = (
                (best.throughput_tok_s - current.throughput_tok_s)
                / current.throughput_tok_s
            )
            metric = (
                f"current BS_max={current.bs_max}, cadence "
                f"{current.cadence_at_capacity_ms:.2f} ms, "
                f"{current.throughput_tok_s:.1f} tok/s (predicted); "
                f"best {','.join(map(str, best.partition))}: "
                f"BS_max={best.bs_max}, cadence "
                f"{best.cadence_at_capacity_ms:.2f} ms, "
                f"{best.throughput_tok_s:.1f} tok/s"
            )
        else:
            gain = (
                (current.bottleneck_ms - best.bottleneck_ms)
                / current.bottleneck_ms
                if current.bottleneck_ms > 0
                else 0.0
            )
            metric = (
                f"current bottleneck {current.bottleneck_ms:.2f} ms; "
                f"best {','.join(map(str, best.partition))}: "
                f"{best.bottleneck_ms:.2f} ms (compute-only objective)"
            )

        current_text = ",".join(map(str, self.current_partition))
        best_text = ",".join(map(str, best.partition))
        if (
            gain * 100.0 > self.args.min_gain_pct
            and not result.keep_current
            and best.partition != self.current_partition
        ):
            detail = (
                f"[advisor] RECOMMEND repartition {current_text} -> {best_text}\n"
                f"  {metric}\n"
                f"  predicted gain {gain:.1%} > --min-gain-pct "
                f"{self.args.min_gain_pct}% and outside the "
                f"{result.noise_sigma:g}-sigma noise band "
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
    parser.add_argument("--interval-s", type=float, default=300.0)
    parser.add_argument("--collect-s", type=float, default=60.0)
    parser.add_argument("--min-gain-pct", type=float, default=8.0)
    parser.add_argument("--t-comm-ms", type=float)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--mamba-full-memory-ratio", type=float, default=2.0)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=32)
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
