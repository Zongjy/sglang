#!/usr/bin/env python3
"""Consume PpStageMetrics (PPM) streams and aggregate a fitting snapshot.

Every PP rank of an SGLang server started with
``SGLANG_ENABLE_PP_STAGE_METRICS=1`` publishes one ``PpStageMetrics`` message
per scheduler iteration on a ``<base_ipc>.pp{r}.dp{d}`` ZMQ PUB endpoint.
This tool subscribes to those endpoints (given directly, or discovered from
the "PPM: ZMQ PUB bound on ..." server log lines), applies the
work-conservation filter, and aggregates per-(rank, log2 batch-size bucket)
Welford statistics.  The resulting snapshot JSON is the input of
``stage_model.StageCostModel.fit``.

Steady-state note: the naive identity ``wall = p2p_wait + cpu + gpu_target +
gpu_draft`` does NOT hold in a concurrent pipeline -- CPU blocking waits
overlap the async GPU forward, so ``wait + gpu`` can exceed ``wall``.  This
tool therefore aggregates a per-iteration service time
``max(gpu_target + gpu_draft, wall - p2p_wait)`` instead of deriving fixed
overheads by subtraction.  ``p2p_wait_ms`` mixes pipeline imbalance with
transfer time and must never be treated as stage service cost.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

PPM_BOUND_LOG = re.compile(r"PPM: ZMQ PUB bound on (\S+)")
# [pp_comm_benchmark] hop 0->1: alpha=0.1234 ms, beta=1.234e-03 ms/token, ...
COMM_BENCHMARK_HOP = re.compile(
    r"\[pp_comm_benchmark\] hop (\d+)->(\d+): "
    r"alpha=([0-9.eE+-]+) ms, beta=([0-9.eE+-]+) ms/token"
)
POLL_TIMEOUT_MS = 100

logger = logging.getLogger("ppm_consumer")


def _load_ppm_codec():
    """Import the runtime PPM schema (adds the repo's python/ to sys.path)."""
    python_dir = str(REPO_ROOT / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    from sglang.srt.observability import pp_stage_metrics

    return pp_stage_metrics


def bucket_of(bs: int) -> int:
    """Log2 bucket floor: 1, 2, 4, 8, ... for bs >= 1."""
    if bs <= 0:
        raise ValueError(f"bucket_of expects bs >= 1, got {bs}")
    return 1 << (bs.bit_length() - 1)


class Welford:
    """Running mean / sample variance / count."""

    __slots__ = ("count", "mean", "_m2")

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    @property
    def var(self) -> float:
        return self._m2 / (self.count - 1) if self.count > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "var": self.var}


@dataclass
class BucketStats:
    """Aggregates of one (pp_rank, bs bucket) cell over kept iterations."""

    bs: Welford = field(default_factory=Welford)
    wall_ms: Welford = field(default_factory=Welford)
    gpu_target_ms: Welford = field(default_factory=Welford)
    gpu_draft_ms: Welford = field(default_factory=Welford)
    p2p_wait_ms: Welford = field(default_factory=Welford)
    accept_len: Welford = field(default_factory=Welford)
    service_ms: Welford = field(default_factory=Welford)
    running_bs: Welford = field(default_factory=Welford)

    @property
    def count(self) -> int:
        return self.wall_ms.count

    def to_dict(self) -> dict[str, Any]:
        wait_fraction = (
            self.p2p_wait_ms.mean / self.wall_ms.mean
            if self.wall_ms.mean > 0
            else 0.0
        )
        return {
            "count": self.count,
            "bs": self.bs.to_dict(),
            "wall_ms": self.wall_ms.to_dict(),
            "gpu_target_ms": self.gpu_target_ms.to_dict(),
            "gpu_draft_ms": self.gpu_draft_ms.to_dict(),
            "p2p_wait_ms": self.p2p_wait_ms.to_dict(),
            "accept_len": self.accept_len.to_dict(),
            "service_ms": self.service_ms.to_dict(),
            "running_bs": self.running_bs.to_dict(),
            "wait_fraction": wait_fraction,
        }


@dataclass
class RankStats:
    """All aggregates for one PP rank."""

    messages_total: int = 0
    messages_kept: int = 0
    idle_messages: int = 0
    filtered_messages: int = 0
    last_seen_s: float | None = None
    idle_interval_ms: Welford = field(default_factory=Welford)
    buckets: dict[int, BucketStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages_total": self.messages_total,
            "messages_kept": self.messages_kept,
            "idle_messages": self.idle_messages,
            "filtered_messages": self.filtered_messages,
            "idle_interval_ms": self.idle_interval_ms.to_dict(),
            "buckets": {
                str(bucket): stats.to_dict()
                for bucket, stats in sorted(self.buckets.items())
            },
        }


class PpmAggregator:
    """Applies the work-conservation filter and aggregates PPM messages.

    An iteration enters the fitting buckets only when it is work-conserving:
    ``num_queued > 0`` (the queue proves offered load exceeds service) or the
    running batch reached the saturation threshold (``saturation_bs``, in
    GLOBAL running-request units -- typically the server's resolved
    ``max_running_requests``).  ``bs == 0`` iterations never enter the
    buckets; they only feed the idle-cadence statistics.
    """

    def __init__(self, saturation_bs: int | None = None) -> None:
        self.saturation_bs = saturation_bs
        self.ranks: dict[int, RankStats] = {}
        self.messages_total = 0
        self.version_mismatches = 0
        self.schema_version = 1
        self.pp_loop_size = 0

    def is_work_conserving(self, metrics: Any) -> bool:
        if metrics.bs <= 0:
            return False
        if metrics.num_queued > 0:
            return True
        if self.saturation_bs is None:
            return False
        # saturation_bs counts GLOBAL running requests, but metrics.bs is the
        # per-slot micro-batch size (running / pp_loop_size); compare against
        # running_bs when the publisher provides it, and fall back to the
        # per-slot comparison only for pre-running_bs snapshot producers.
        running_bs = int(getattr(metrics, "running_bs", 0) or 0)
        if running_bs > 0:
            return running_bs >= self.saturation_bs
        return metrics.bs >= self.saturation_bs

    def observe(self, metrics: Any) -> bool:
        """Record one decoded PpStageMetrics; returns True if it was kept."""
        self.messages_total += 1
        version = int(getattr(metrics, "version", 1))
        self.schema_version = max(self.schema_version, version)
        loop_size = int(getattr(metrics, "pp_loop_size", 0) or 0)
        if loop_size > 0:
            self.pp_loop_size = max(self.pp_loop_size, loop_size)
        rank = self.ranks.setdefault(metrics.pp_rank, RankStats())
        rank.messages_total += 1

        now_s = time.monotonic()
        if metrics.bs == 0:
            rank.idle_messages += 1
            if rank.last_seen_s is not None:
                rank.idle_interval_ms.add((now_s - rank.last_seen_s) * 1000.0)
            rank.last_seen_s = now_s
            return False
        rank.last_seen_s = now_s

        if not self.is_work_conserving(metrics):
            rank.filtered_messages += 1
            return False

        bucket = bucket_of(metrics.bs)
        stats = rank.buckets.setdefault(bucket, BucketStats())
        stats.bs.add(float(metrics.bs))
        stats.wall_ms.add(metrics.wall_ms)
        stats.gpu_target_ms.add(metrics.gpu_target_ms)
        stats.gpu_draft_ms.add(metrics.gpu_draft_ms)
        stats.p2p_wait_ms.add(metrics.p2p_wait_ms)
        stats.accept_len.add(metrics.accept_len)
        # Per-iteration service time: CPU blocking waits overlap the async GPU
        # forward in a concurrent pipeline, so wall - p2p_wait can undershoot
        # the GPU busy time; take the per-message max of the two views.
        stats.service_ms.add(
            max(
                metrics.gpu_target_ms + metrics.gpu_draft_ms,
                metrics.wall_ms - metrics.p2p_wait_ms,
            )
        )
        stats.running_bs.add(float(getattr(metrics, "running_bs", 0) or 0))
        rank.messages_kept += 1
        return True

    def snapshot(self, endpoints: Sequence[str], duration_s: float) -> dict[str, Any]:
        return {
            "version": 1,
            "schema_version": self.schema_version,
            "pp_loop_size": self.pp_loop_size,
            "duration_s": duration_s,
            "endpoints": list(endpoints),
            "saturation_bs": self.saturation_bs,
            "messages_total": self.messages_total,
            "messages_kept": sum(rank.messages_kept for rank in self.ranks.values()),
            "version_mismatches": self.version_mismatches,
            "ranks": {
                str(rank): stats.to_dict()
                for rank, stats in sorted(self.ranks.items())
            },
        }


def parse_ppm_endpoints(text: str) -> list[str]:
    """Extract PPM PUB endpoints from server log content."""
    endpoints: list[str] = []
    for match in PPM_BOUND_LOG.finditer(text):
        endpoint = match.group(1)
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    return endpoints


def parse_comm_benchmark_hops(text: str) -> dict[str, tuple[float, float]]:
    """Parse per-hop (alpha_ms, beta_ms_per_token) fits from a server log.

    The runtime startup micro-benchmark (SGLANG_PP_COMM_BENCHMARK=1,
    ``python/sglang/srt/distributed/pp_comm_benchmark.py``) logs one line per
    hop: ``[pp_comm_benchmark] hop 0->1: alpha=... ms, beta=... ms/token,
    effective_bw=... GB/s, points=[...]``.
    """
    hops: dict[str, tuple[float, float]] = {}
    for match in COMM_BENCHMARK_HOP.finditer(text):
        hop = f"{match.group(1)}->{match.group(2)}"
        hops[hop] = (float(match.group(3)), float(match.group(4)))
    return hops


def t_comm_from_log(text: str, num_tokens: float) -> float | None:
    """Worst-hop one-way transfer time t_hop = alpha + beta * num_tokens.

    ``num_tokens`` is the per-hop transfer size at the working point
    (verify moves bs x block_size hidden states per hop).  Returns None when
    the log has no comm benchmark lines.
    """
    hops = parse_comm_benchmark_hops(text)
    if not hops:
        return None
    return max(alpha + beta * num_tokens for alpha, beta in hops.values())


def collect(
    endpoints: Sequence[str],
    duration_s: float | None = None,
    max_messages: int | None = None,
    saturation_bs: int | None = None,
    context: Any = None,
) -> dict[str, Any]:
    """Subscribe to PPM endpoints and aggregate until a stop condition.

    Stops when ``duration_s`` elapsed or ``max_messages`` were received;
    at least one condition must be given.
    """
    if duration_s is None and max_messages is None:
        raise ValueError("collect needs duration_s and/or max_messages.")
    if not endpoints:
        raise ValueError("collect needs at least one endpoint.")

    import zmq

    ppm = _load_ppm_codec()
    ctx = context or zmq.Context()
    sockets = []
    poller = zmq.Poller()
    for endpoint in endpoints:
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 100_000)
        sock.connect(endpoint)
        sockets.append(sock)
        poller.register(sock, zmq.POLLIN)

    aggregator = PpmAggregator(saturation_bs=saturation_bs)
    started_s = time.monotonic()
    warned_version = False
    try:
        while True:
            elapsed = time.monotonic() - started_s
            if duration_s is not None and elapsed >= duration_s:
                break
            if max_messages is not None and aggregator.messages_total >= max_messages:
                break
            remaining_ms = POLL_TIMEOUT_MS
            if duration_s is not None:
                remaining_ms = max(
                    1, min(POLL_TIMEOUT_MS, int((duration_s - elapsed) * 1000))
                )
            events = dict(poller.poll(remaining_ms))
            for sock, _ in events.items():
                frames = sock.recv_multipart()
                payload = frames[-1]
                metrics = ppm.decode(payload)
                if metrics.version != ppm.PPM_VERSION:
                    aggregator.version_mismatches += 1
                    if not warned_version:
                        logger.warning(
                            "PPM schema version %d != consumer %d; decoding anyway",
                            metrics.version,
                            ppm.PPM_VERSION,
                        )
                        warned_version = True
                aggregator.observe(metrics)
    finally:
        for sock in sockets:
            sock.close(linger=0)
        if context is None:
            ctx.term()

    if aggregator.schema_version < 2:
        logger.warning(
            "PPM schema v%d snapshot: bs is the per-slot microbatch size; "
            "running_bs is unknown, pp_loop_size falls back to pp_size",
            aggregator.schema_version,
        )
    return aggregator.snapshot(endpoints, time.monotonic() - started_s)


def resolve_endpoints(args: argparse.Namespace) -> list[str]:
    endpoints: list[str] = list(args.endpoint or [])
    for log_path in args.server_log or []:
        try:
            text = Path(log_path).read_text(errors="replace")
        except OSError as exc:
            raise SystemExit(f"cannot read server log {log_path}: {exc}") from exc
        for endpoint in parse_ppm_endpoints(text):
            if endpoint not in endpoints:
                endpoints.append(endpoint)
    if not endpoints:
        raise SystemExit(
            "no PPM endpoints: pass --endpoint and/or --server-log with "
            "'PPM: ZMQ PUB bound on' lines"
        )
    return endpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        action="append",
        help="PPM PUB endpoint (repeatable), e.g. ipc:///tmp/xxx.pp0.dp0",
    )
    parser.add_argument(
        "--server-log",
        action="append",
        help="server log to scan for 'PPM: ZMQ PUB bound on' lines (repeatable)",
    )
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--max-messages", type=int)
    parser.add_argument(
        "--saturation-bs",
        type=int,
        help=(
            "batch size treated as saturating for the work-conservation "
            "filter (typically max_running_requests); default: only "
            "num_queued > 0 iterations are kept"
        ),
    )
    parser.add_argument("-o", "--output", type=Path, help="snapshot JSON output path")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[ppm_consumer] %(message)s",
    )
    endpoints = resolve_endpoints(args)
    logger.info("subscribing to %d endpoint(s):", len(endpoints))
    for endpoint in endpoints:
        logger.info("  %s", endpoint)
    snapshot = collect(
        endpoints,
        duration_s=args.duration_s,
        max_messages=args.max_messages,
        saturation_bs=args.saturation_bs,
    )
    text = json.dumps(snapshot, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        logger.info(
            "wrote %s (%d messages, %d kept)",
            args.output,
            snapshot["messages_total"],
            snapshot["messages_kept"],
        )
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
