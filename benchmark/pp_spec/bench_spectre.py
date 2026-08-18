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
import random
import time
from dataclasses import asdict, dataclass, field
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
    request_index: int
    scheduled_offset_s: float
    dispatch_offset_s: float
    ttft_s: float | None
    e2e_s: float | None
    completion_tokens: int | None
    prompt_tokens: int | None
    accept_length: float | None
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
    tpot_p50_s: float | None
    mean_accept_length: float | None
    peak_active: int
    max_queue: int
    arrivals_at_cap: int


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


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[idx]


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
    ttft = None
    accept_length = None
    completion_tokens = None
    prompt_tokens = None
    error = None
    started = time.perf_counter()
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
                if ttft is None and chunk.get("text"):
                    ttft = time.perf_counter() - benchmark_started
                meta = chunk.get("meta_info") or {}
                if meta.get("finish_reason") is not None:
                    completion_tokens = meta.get("completion_tokens")
                    prompt_tokens = meta.get("prompt_tokens")
                    accept_length = meta.get("spec_accept_length")
    except Exception as exc:  # noqa: BLE001 - record per-request errors
        error = repr(exc)
    done = time.perf_counter()
    return RequestResult(
        request_index=request_index,
        scheduled_offset_s=scheduled_offset,
        dispatch_offset_s=started - benchmark_started,
        ttft_s=ttft,
        e2e_s=done - benchmark_started if ttft is not None else None,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        accept_length=accept_length,
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
        (r.e2e_s for r in results if r.e2e_s is not None),
        default=time.perf_counter() - benchmark_started,
    )
    good = [r for r in results if r.error is None]
    total_tokens = sum(r.completion_tokens or 0 for r in good)
    ttfts = [r.ttft_s - r.dispatch_offset_s for r in good if r.ttft_s is not None]
    tpots = [
        (r.e2e_s - r.ttft_s) / (r.completion_tokens - 1)
        for r in good
        if r.e2e_s is not None
        and r.ttft_s is not None
        and (r.completion_tokens or 0) > 1
    ]
    accepts = [r.accept_length for r in good if r.accept_length is not None]
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
        tpot_p50_s=percentile(tpots, 0.5),
        mean_accept_length=(
            sum(accepts) / len(accepts) if accepts else None
        ),
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
