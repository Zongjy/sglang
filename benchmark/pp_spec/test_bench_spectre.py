from __future__ import annotations

import pytest

from benchmark.pp_spec.bench_spectre import (
    RequestMetricState,
    calculate_accept_length,
    calculate_tpot,
    percentile,
)


def test_speculative_bursts_use_first_and_last_token_frames() -> None:
    state = RequestMetricState()

    state.observe({"completion_tokens": 4, "prompt_tokens": 32}, observed_s=0.1)
    state.observe({"completion_tokens": 8}, observed_s=0.5)
    state.observe(
        {
            "completion_tokens": 8,
            "finish_reason": {"type": "length"},
            "spec_accept_length": 8 / 3,
            "spec_verify_ct": 3,
            "spec_num_correct_drafts": 17,
        },
        observed_s=0.8,
    )

    assert state.validation_error() is None
    assert state.first_token_s == pytest.approx(0.1)
    assert state.last_token_s == pytest.approx(0.5)
    assert state.tpot_s == pytest.approx(0.4 / 7)
    assert state.accept_length == pytest.approx(20 / 3)
    assert state.server_accept_length == pytest.approx(8 / 3)


def test_completion_token_count_must_be_monotonic() -> None:
    state = RequestMetricState()
    state.observe({"completion_tokens": 8}, observed_s=0.1)

    with pytest.raises(ValueError, match="completion_tokens decreased"):
        state.observe({"completion_tokens": 7}, observed_s=0.2)


def test_speculative_counters_must_be_monotonic() -> None:
    state = RequestMetricState()
    state.observe(
        {"spec_verify_ct": 4, "spec_num_correct_drafts": 20}, observed_s=0.1
    )

    with pytest.raises(ValueError, match="spec_verify_ct decreased"):
        state.observe({"spec_verify_ct": 3}, observed_s=0.2)

    with pytest.raises(ValueError, match="spec_num_correct_drafts decreased"):
        state.observe({"spec_num_correct_drafts": 19}, observed_s=0.2)


def test_accept_length_uses_verify_weighted_raw_counters() -> None:
    # One request has 8.0 accept length over one verify; another has 1.0 over
    # nine verifies. The workload-wide value is 1.7, not request-mean 4.5.
    assert calculate_accept_length(1 + 9, 7 + 0) == pytest.approx(1.7)
    assert calculate_accept_length(0, 0) is None


def test_tpot_excludes_first_token_and_stream_close() -> None:
    assert calculate_tpot(0.2, 2.75, 256) == pytest.approx(0.01)
    assert calculate_tpot(0.2, 100.0, 1) is None


def test_percentile_uses_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) == pytest.approx(2.5)
    assert percentile(values, 0.99) == pytest.approx(3.97)
    assert percentile([], 0.99) is None

    with pytest.raises(ValueError, match="must be in"):
        percentile(values, 1.01)
