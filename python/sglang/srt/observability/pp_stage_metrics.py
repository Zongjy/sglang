"""
Per-iteration per-PP-rank stage metrics (PPM) for pipeline-parallel telemetry.

One message per scheduler iteration per PP rank, published over ZMQ PUB so
external consumers (the PP partition tuner / advisor) can fit per-stage cost
models in real time without torch-profiler traces.

Every PP rank publishes its own stage decomposition on its own endpoint;
unlike ForwardPassMetrics this stream is NOT shared with Dynamo, so the
schema is free to evolve (bump ``PPM_VERSION`` on incompatible changes).

Data flow::

    event_loop_pp (each PP rank, attn_tp_rank == 0)
        SchedulerMetricsReporter.emit_pp_stage_metrics()
          -> _PpmPublisherThread -> ZMQ PUB (localhost)

    External consumer:
        ZMQ SUB -> deserialize PpStageMetrics
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from itertools import count

import msgspec

# Schema version. Bump when the schema changes incompatibly.
# v2: added running_bs (global running requests across all PP microbatch
# slots) and pp_loop_size.  ``bs`` remains the per-slot microbatch size.
PPM_VERSION: int = 2

logger = logging.getLogger(__name__)


class PpStageMetrics(
    msgspec.Struct,
    frozen=True,
    gc=False,
):
    """Per-iteration stage timing for one PP rank.

    All durations are milliseconds. ``gpu_target_ms`` / ``gpu_draft_ms`` come
    from DeviceTimer (CUDA events, harvested without forcing a sync) and may
    lag the iteration that produced them by one harvest cycle; they are exact
    in aggregate. ``p2p_wait_ms`` is wall time blocked on PP recvs — it mixes
    pipeline imbalance (bubble) with transfer time and must never be treated
    as stage service cost. ``accept_len`` is the mean speculative accept
    length over this iteration (last rank only; 0 elsewhere).

    ``bs`` is the PER-SLOT microbatch size: the PP loop has ``pp_loop_size``
    microbatch slots and the global running requests are spread across them,
    so the global running batch is roughly ``bs * pp_loop_size``.
    ``running_bs`` (v2) is the exact global running-request count summed
    over all slots.
    """

    version: int = PPM_VERSION
    worker_id: str = ""
    pp_rank: int = 0
    dp_rank: int = 0
    counter_id: int = 0
    wall_ms: float = 0.0
    bs: int = 0
    num_queued: int = 0
    gpu_target_ms: float = 0.0
    gpu_draft_ms: float = 0.0
    p2p_wait_ms: float = 0.0
    accept_len: float = 0.0
    running_bs: int = 0
    pp_loop_size: int = 0


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(PpStageMetrics)


def encode(metrics: PpStageMetrics) -> bytes:
    return _encoder.encode(metrics)


def decode(data: bytes) -> PpStageMetrics:
    return _decoder.decode(data)


class _PpmPublisherThread:
    """Background thread that serializes and sends PpStageMetrics over ZMQ.

    Same fire-and-forget pattern as the FPM publisher: a bounded queue with
    drop-on-full so a stalled consumer can never back-pressure the scheduler
    loop.
    """

    SHUTDOWN_TIMEOUT: float = 1.0

    def __init__(
        self,
        endpoint: str,
        worker_id: str,
        pp_rank: int,
        dp_rank: int,
        max_queue_size: int = 10_000,
    ) -> None:
        import zmq

        self._queue: queue.Queue[PpStageMetrics | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._seq = count()
        self._worker_id = worker_id
        self._pp_rank = pp_rank
        self._dp_rank = dp_rank

        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(endpoint)
        self._zmq = zmq

        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ppm-zmq-publisher"
        )
        self._thread.start()

    def publish(self, metrics: PpStageMetrics) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(metrics)
        except queue.Full:
            pass

    def shutdown(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            self._pub.close(linger=0)
            self._ctx.term()
        except Exception:
            pass

    def _run(self) -> None:
        zmq = self._zmq
        topic = b""

        while self._running or not self._queue.empty():
            try:
                metrics = self._queue.get(timeout=0.5)
                if metrics is None:
                    break
            except queue.Empty:
                continue

            try:
                seq = next(self._seq)
                metrics = msgspec.structs.replace(
                    metrics,
                    counter_id=seq,
                    worker_id=self._worker_id,
                    pp_rank=self._pp_rank,
                    dp_rank=self._dp_rank,
                )
                payload = encode(metrics)
                seq_bytes = seq.to_bytes(8, "big")
                self._pub.send_multipart((topic, seq_bytes, payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception:
                # Once: a persistent failure (e.g. unencodable field) would
                # otherwise flood the scheduler log with one traceback per
                # iteration.
                logger.warning_once("PPM publisher send failed")
