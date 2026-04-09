"""
policy_ewma.py — AIMD with an EWMA latency slope guard (H5 baseline, no sklearn)
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import numpy as np

import config
from metrics import summarize_simulation_episode
from sim_core import StepRecord, aimd_adjust_concurrency, set_seed, simulate_step, update_rolling_p95


def simulate_aimd_episode_with_ewma_latency_guard(
    num_steps: int,
    *,
    ewma_alpha: float = 0.25,
    slope_block_ms: float = 0.35,
) -> Tuple[List[StepRecord], dict]:
    """AIMD plus a cheap latency trend check before additive increases."""
    set_seed(config.RANDOM_SEED)
    rolling_request_latencies_ms: Deque[float] = deque(maxlen=config.LAT_ROLLING_MAXLEN)
    current_concurrency = 1
    episode_step_records: List[StepRecord] = []
    smoothed_latency_ms: float | None = None
    previous_smoothed_latency_ms: float | None = None

    for step_index in range(num_steps):
        current_step_record = simulate_step(step_index, current_concurrency)
        current_step_record.latency_p95_recent = update_rolling_p95(
            rolling_request_latencies_ms, current_step_record.latency_ms_list
        )
        episode_step_records.append(current_step_record)

        batch_mean_latency_ms = (
            float(np.mean(current_step_record.latency_ms_list))
            if current_step_record.latency_ms_list
            else 0.0
        )
        if smoothed_latency_ms is None:
            smoothed_latency_ms = batch_mean_latency_ms
        else:
            smoothed_latency_ms = (
                ewma_alpha * batch_mean_latency_ms + (1.0 - ewma_alpha) * smoothed_latency_ms
            )

        aimd_proposed_concurrency = aimd_adjust_concurrency(
            current_concurrency,
            current_step_record.had_any_429,
            config.AIMD_ADD_ON_SUCCESS,
            config.AIMD_MULT_ON_429,
        )

        if current_step_record.had_any_429:
            current_concurrency = aimd_proposed_concurrency
        elif aimd_proposed_concurrency > current_concurrency:
            latency_slope_blocks_increase = (
                previous_smoothed_latency_ms is not None
                and (smoothed_latency_ms - previous_smoothed_latency_ms) > slope_block_ms
            )
            if latency_slope_blocks_increase:
                pass
            else:
                current_concurrency = aimd_proposed_concurrency
        else:
            current_concurrency = aimd_proposed_concurrency

        previous_smoothed_latency_ms = smoothed_latency_ms

    return episode_step_records, summarize_simulation_episode(episode_step_records)
