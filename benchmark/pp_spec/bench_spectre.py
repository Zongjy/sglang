#!/usr/bin/env python3
"""SPECTRE-style online benchmark for sglang (arXiv 2605.08151 methodology).

Open-loop Poisson arrivals with a client-side FIFO ``max_concurrency`` cap
(effective-batch-size control), ShareGPT prompts truncated to a fixed token
budget, greedy decoding with fixed-length output and ignore_eos. Reports
throughput, TTFT/TPOT percentiles and speculative accept length per load
point. Mirrors the mini-sgl online.py dispatcher semantics.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "data" / "sharegpt.json"
DEFAULT_LOAD_POINTS = "1:1:8,4:2:16,8:4:32,16:8:64,32:16:128"


@dataclass(frozen=True)
class LoadPoint:
    max_concurrency: int
    request_rate_rps: float
    num_requests: int


def parse_load_points(text: str) -> list[LoadPoint]:
    points = []
    for chunk in text.split(","):
        fields = chunk.strip().split(":")
        if len(fields) != 3:
            raise ValueError(f"bad load point {chunk!r}; expected c:qps:n")
        points.append(LoadPoint(int(fields[0]), float(fields[1]), int(fields[2])))
    return points


@dataclass
class RequestResult:
    # ttft/e2e/stream_close are request-relative and e2e ends at the last token.
    # completion_offset_s is benchmark-relative and drives measurement duration.
    request_index: int
    scheduled_offset_s: float
    dispatch_offset_s: float
    ttft_s: float | None
    e2e_s: float | None
    completion_offset_s: float | None
    stream_close_s: float
    tpot_s: float | None
    completion_tokens: int | None
    prompt_tokens: int | None
    accept_length: float | None
    server_accept_length: float | None
    spec_verify_ct: int | None
    spec_num_correct_drafts: int | None
    error: str | None = None


@dataclass
class PointSummary:
    label: str
    max_concurrency: int
    request_rate_rps: float
    num_requests: int
    completed: int
    measurement_duration_s: float
    output_throughput_tok_s: float
    request_throughput_req_s: float
    ttft_p50_s: float | None
    ttft_p99_s: float | None
    mean_tpot_s: float | None
    tpot_p50_s: float | None
    tpot_p90_s: float | None
    tpot_p99_s: float | None
    mean_accept_length: float | None
    request_mean_accept_length: float | None
    accept_length_p50: float | None
    accept_length_p90: float | None
    accept_length_p99: float | None
    spec_metric_requests: int
    total_spec_verify_ct: int
    total_spec_num_correct_drafts: int
    peak_active: int
    max_queue: int
    arrivals_at_cap: int


@dataclass
class RequestMetricState:
    """Accumulate token timing and speculative counters from cumulative SSE metadata."""

    first_token_s: float | None = None
    last_token_s: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    server_accept_length: float | None = None
    spec_verify_ct: int | None = None
    spec_num_correct_drafts: int | None = None
    finished: bool = False

    def observe(self, meta: dict, observed_s: float) -> None:
        completion_tokens = _optional_nonnegative_int(
            meta.get("completion_tokens"), "completion_tokens"
        )
        if completion_tokens is not None:
            previous = self.completion_tokens or 0
            if completion_tokens < previous:
                raise ValueError(
                    "completion_tokens decreased from "
                    f"{previous} to {completion_tokens}"
                )
            if completion_tokens > previous:
                if self.first_token_s is None:
                    self.first_token_s = observed_s
                self.last_token_s = observed_s
            self.completion_tokens = completion_tokens

        prompt_tokens = _optional_nonnegative_int(
            meta.get("prompt_tokens"), "prompt_tokens"
        )
        if prompt_tokens is not None:
            self.prompt_tokens = prompt_tokens

        spec_verify_ct = _optional_nonnegative_int(
            meta.get("spec_verify_ct"), "spec_verify_ct"
        )
        if spec_verify_ct is not None:
            _ensure_monotonic(
                self.spec_verify_ct, spec_verify_ct, "spec_verify_ct"
            )
            self.spec_verify_ct = spec_verify_ct

        spec_num_correct_drafts = _optional_nonnegative_int(
            meta.get("spec_num_correct_drafts"), "spec_num_correct_drafts"
        )
        if spec_num_correct_drafts is not None:
            _ensure_monotonic(
                self.spec_num_correct_drafts,
                spec_num_correct_drafts,
                "spec_num_correct_drafts",
            )
            self.spec_num_correct_drafts = spec_num_correct_drafts

        server_accept_length = meta.get("spec_accept_length")
        if server_accept_length is not None:
            self.server_accept_length = float(server_accept_length)

        if meta.get("finish_reason") is not None:
            self.finished = True

    @property
    def accept_length(self) -> float | None:
        return calculate_accept_length(
            self.spec_verify_ct, self.spec_num_correct_drafts
        )

    @property
    def tpot_s(self) -> float | None:
        return calculate_tpot(
            self.first_token_s, self.last_token_s, self.completion_tokens
        )

    def validation_error(self) -> str | None:
        if not self.finished:
            return "stream ended without finish_reason"
        if self.completion_tokens is None:
            return "finished response has no completion_tokens"
        if self.completion_tokens > 0 and self.first_token_s is None:
            return "finished response has tokens but no token arrival timestamp"
        return None


def load_sharegpt_prompts(
    path: Path, limit: int, max_prompt_tokens: int, tokenizer
) -> list[str]:
    with open(path) as handle:
        data = json.load(handle)
    prompts = []
    for row in data:
        turns = row.get("conversations") or []
        if not turns or turns[0].get("from") != "human":
            continue
        text = turns[0]["value"].strip()
        if len(text) < 32:
            continue
        ids = tokenizer.encode(text)
        if len(ids) > max_prompt_tokens:
            ids = ids[:max_prompt_tokens]
            text = tokenizer.decode(ids)
        prompts.append(text)
        if len(prompts) >= limit:
            break
    if len(prompts) < limit:
        raise RuntimeError(f"only {len(prompts)} usable ShareGPT prompts, need {limit}")
    return prompts


def build_synthetic_prompts(
    limit: int, max_prompt_tokens: int, tokenizer
) -> list[str]:
    """Build deterministic, unique prompts without an external dataset."""
    seed_text = (
        "Analyze the system behavior carefully and continue the technical report "
        "with concrete observations about latency, throughput, and reliability. "
    )
    prompts = []
    for index in range(limit):
        text = f"Synthetic request {index}. " + seed_text * (max_prompt_tokens // 12 + 2)
        ids = tokenizer.encode(text)[:max_prompt_tokens]
        prompts.append(tokenizer.decode(ids))
    return prompts


def _optional_nonnegative_int(value, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")
    return value


def _ensure_monotonic(previous: int | None, current: int, field_name: str) -> None:
    if previous is not None and current < previous:
        raise ValueError(f"{field_name} decreased from {previous} to {current}")


def calculate_accept_length(
    spec_verify_ct: int | None, spec_num_correct_drafts: int | None
) -> float | None:
    """Return committed tokens per verify from the untruncated spec counters."""
    if not spec_verify_ct or spec_num_correct_drafts is None:
        return None
    return (spec_num_correct_drafts + spec_verify_ct) / spec_verify_ct


def calculate_tpot(
    first_token_s: float | None,
    last_token_s: float | None,
    completion_tokens: int | None,
) -> float | None:
    """Return per-request mean time between output tokens."""
    if (
        first_token_s is None
        or last_token_s is None
        or completion_tokens is None
        or completion_tokens <= 1
    ):
        return None
    return (last_token_s - first_token_s) / (completion_tokens - 1)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError(f"percentile q must be in [0, 1], got {q}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def request_once(
    client: httpx.AsyncClient,
    prompt: str,
    request_index: int,
    scheduled_offset: float,
    benchmark_started: float,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    metrics = RequestMetricState()
    error = None
    started = time.perf_counter()
    dispatch_offset_s = started - benchmark_started
    try:
        async with client.stream("POST", "/generate", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line.removeprefix("data:").strip()
                if not raw or raw == "[DONE]":
                    continue
                chunk = json.loads(raw)
                meta = chunk.get("meta_info") or {}
                metrics.observe(meta, time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - record per-request errors
        error = repr(exc)
    stream_close_s = time.perf_counter() - started
    if error is None:
        error = metrics.validation_error()
    completion_offset_s = (
        dispatch_offset_s + metrics.last_token_s
        if error is None and metrics.last_token_s is not None
        else None
    )
    return RequestResult(
        request_index=request_index,
        scheduled_offset_s=scheduled_offset,
        dispatch_offset_s=dispatch_offset_s,
        ttft_s=metrics.first_token_s,
        e2e_s=metrics.last_token_s,
        completion_offset_s=completion_offset_s,
        stream_close_s=stream_close_s,
        tpot_s=metrics.tpot_s,
        completion_tokens=metrics.completion_tokens,
        prompt_tokens=metrics.prompt_tokens,
        accept_length=metrics.accept_length,
        server_accept_length=metrics.server_accept_length,
        spec_verify_ct=metrics.spec_verify_ct,
        spec_num_correct_drafts=metrics.spec_num_correct_drafts,
        error=error,
    )


async def run_point(
    config: argparse.Namespace,
    point: LoadPoint,
    prompts: list[str],
) -> tuple[list[RequestResult], PointSummary]:
    rng = random.Random(config.seed)
    arrival_offsets = []
    next_at = 0.0
    for _ in range(point.num_requests):
        next_at += rng.expovariate(point.request_rate_rps)
        arrival_offsets.append(next_at)

    client_count = min(point.max_concurrency, point.num_requests)
    queue: asyncio.Queue = asyncio.Queue()
    results: list[RequestResult] = []
    active = peak_active = max_queue = arrivals_at_cap = 0
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    timeout = httpx.Timeout(config.request_timeout_s)
    clients = [
        httpx.AsyncClient(base_url=config.url, timeout=timeout, limits=limits)
        for _ in range(client_count)
    ]
    benchmark_started = time.perf_counter()
    try:

        async def worker(client: httpx.AsyncClient) -> None:
            nonlocal active, peak_active
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                prompt, request_index, scheduled = item
                active += 1
                peak_active = max(peak_active, active)
                try:
                    results.append(
                        await request_once(
                            client,
                            prompt,
                            request_index,
                            scheduled,
                            benchmark_started,
                            config.max_tokens,
                            config.temperature,
                        )
                    )
                    if len(results) % 16 == 0 or len(results) == point.num_requests:
                        print(
                            f"[{config.label} C={point.max_concurrency} "
                            f"QPS={point.request_rate_rps:g}] "
                            f"{len(results)}/{point.num_requests}",
                            flush=True,
                        )
                finally:
                    active -= 1
                    queue.task_done()

        tasks = [asyncio.create_task(worker(client)) for client in clients]
        for request_index, (prompt, offset) in enumerate(
            zip(prompts[: point.num_requests], arrival_offsets, strict=True)
        ):
            delay = benchmark_started + offset - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            queue_depth = queue.qsize()
            if active >= point.max_concurrency or queue_depth > 0:
                arrivals_at_cap += 1
            max_queue = max(max_queue, queue_depth)
            queue.put_nowait((prompt, request_index, offset))
        for _ in range(client_count):
            queue.put_nowait(None)
        await queue.join()
        await asyncio.gather(*tasks)
    finally:
        await asyncio.gather(*(client.aclose() for client in clients), return_exceptions=True)

    results.sort(key=lambda r: r.request_index)
    duration = max(
        (
            r.completion_offset_s
            for r in results
            if r.completion_offset_s is not None
        ),
        default=time.perf_counter() - benchmark_started,
    )
    good = [r for r in results if r.error is None]
    total_tokens = sum(r.completion_tokens or 0 for r in good)
    ttfts = [r.ttft_s for r in good if r.ttft_s is not None]
    tpots = [r.tpot_s for r in good if r.tpot_s is not None]
    spec_results = [
        r
        for r in good
        if r.spec_verify_ct is not None
        and r.spec_verify_ct > 0
        and r.spec_num_correct_drafts is not None
    ]
    total_spec_verify_ct = sum(r.spec_verify_ct or 0 for r in spec_results)
    total_spec_num_correct_drafts = sum(
        r.spec_num_correct_drafts or 0 for r in spec_results
    )
    accepts = [
        r.accept_length for r in spec_results if r.accept_length is not None
    ]
    summary = PointSummary(
        label=config.label,
        max_concurrency=point.max_concurrency,
        request_rate_rps=point.request_rate_rps,
        num_requests=point.num_requests,
        completed=len(good),
        measurement_duration_s=duration,
        output_throughput_tok_s=total_tokens / duration if duration > 0 else 0.0,
        request_throughput_req_s=len(good) / duration if duration > 0 else 0.0,
        ttft_p50_s=percentile(ttfts, 0.5),
        ttft_p99_s=percentile(ttfts, 0.99),
        mean_tpot_s=sum(tpots) / len(tpots) if tpots else None,
        tpot_p50_s=percentile(tpots, 0.5),
        tpot_p90_s=percentile(tpots, 0.9),
        tpot_p99_s=percentile(tpots, 0.99),
        mean_accept_length=calculate_accept_length(
            total_spec_verify_ct, total_spec_num_correct_drafts
        ),
        request_mean_accept_length=(
            sum(accepts) / len(accepts) if accepts else None
        ),
        accept_length_p50=percentile(accepts, 0.5),
        accept_length_p90=percentile(accepts, 0.9),
        accept_length_p99=percentile(accepts, 0.99),
        spec_metric_requests=len(spec_results),
        total_spec_verify_ct=total_spec_verify_ct,
        total_spec_num_correct_drafts=total_spec_num_correct_drafts,
        peak_active=peak_active,
        max_queue=max_queue,
        arrivals_at_cap=arrivals_at_cap,
    )
    return results, summary


async def warm_up(config: argparse.Namespace, prompt: str) -> None:
    timeout = httpx.Timeout(config.request_timeout_s)
    async with httpx.AsyncClient(base_url=config.url, timeout=timeout) as client:
        for _ in range(4):
            resp = await client.post(
                "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 32,
                        "ignore_eos": True,
                    },
                },
            )
            resp.raise_for_status()
    await asyncio.sleep(config.cooldown_s)


async def main_async(config: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer, trust_remote_code=True
    )
    total_requests = max(p.num_requests for p in config.points)
    if config.synthetic:
        prompts = build_synthetic_prompts(
            total_requests, config.prompt_max_tokens, tokenizer
        )
    else:
        prompts = load_sharegpt_prompts(
            Path(config.dataset), total_requests, config.prompt_max_tokens, tokenizer
        )

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    await warm_up(config, prompts[0])

    summaries: list[PointSummary] = []
    for point in config.points:
        print(f"=== {config.label} C={point.max_concurrency} QPS={point.request_rate_rps} ===", flush=True)
        results, summary = await run_point(config, point, prompts)
        summaries.append(summary)
        with open(out_dir / f"{config.label}_c{point.max_concurrency}.jsonl", "w") as handle:
            for result in results:
                handle.write(json.dumps(asdict(result)) + "\n")
        print(json.dumps(asdict(summary), indent=2), flush=True)
        await asyncio.sleep(config.cooldown_s)

    with open(out_dir / f"{config.label}_summary.json", "w") as handle:
        json.dump([asdict(s) for s in summaries], handle, indent=2)
    with open(out_dir / f"{config.label}_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))
    print(f"saved summaries to {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="sglang server base url")
    parser.add_argument("--label", required=True, help="run label, e.g. pp2_dflash")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use deterministic generated prompts instead of a dataset",
    )
    parser.add_argument(
        "--tokenizer", default="Qwen/Qwen3.6-27B"
    )
    parser.add_argument("--load-points", default=DEFAULT_LOAD_POINTS)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--cooldown-s", type=float, default=10.0)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    config = parser.parse_args()
    config.points = parse_load_points(config.load_points)
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
