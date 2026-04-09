"""
metrics.py

Episode-level **summaries** for the rate-limit simulator (counts, rates, concurrency
statistics), **pretty printing** for A/B comparisons between two policy runs, and a tiny
**micro-benchmark** (H4) for the cost of one classifier probability evaluation on the
decision path.

This module depends on ``features.classifier_predict_probability_batch_had_429`` so the
benchmark exercises the *same* code path policies use, not raw ``model.predict_proba``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

import config
from features import classifier_predict_probability_batch_had_429
from sim_core import StepRecord


def summarize_simulation_episode(records: List[StepRecord]) -> Dict[str, float]:
    """
    Reduce a full list of per-step ``StepRecord`` objects to a small dict of headline
    scalars you can log, plot, or compare across policies.

    Keys (all values are ``float`` for easy JSON / CSV export):

    - **total_429_requests:** Sum of ``num_429`` across steps (count of simulated
      requests that received HTTP 429).
    - **total_requests:** Sum of ``num_ok + num_429`` across steps (all simulated HTTP
      attempts in the episode).
    - **429_rate:** ``total_429_requests / total_requests`` (0 if there were no
      requests).
    - **mean_concurrency:** Average of ``concurrency`` over steps (how aggressively the
      client kept parallelism on average).
    - **final_concurrency:** ``concurrency`` on the last step (end-state probe level).
    - **frac_near_true_capacity:** Fraction of steps where ``concurrency`` is within ±2 of
      ``config.TRUE_CAPACITY``. This uses the simulator’s *hidden* capacity as an oracle
      only for **evaluation**—a real client would not know ``TRUE_CAPACITY``.

    Args:
        records:
            Chronological episode; must be non-empty if you rely on ``final_concurrency``
            (otherwise indexing ``records[-1]`` will raise).

    Returns:
        Dictionary mapping metric name strings to floats as described above.

    Notes:
        * These are **simulator** metrics, not production SLOs.
        * Near-capacity fraction is a deliberately loose ±2 band; tighten or replace if
          you care about exact convergence to the plateau.
    """
    total_rate_limited_requests = sum(step.num_429 for step in records)
    total_simulated_requests = sum(step.num_ok + step.num_429 for step in records)
    num_episode_steps = len(records)
    fraction_steps_near_hidden_capacity = sum(
        1 for step in records if abs(step.concurrency - config.TRUE_CAPACITY) <= 2
    ) / max(1, num_episode_steps)
    return {
        "total_429_requests": float(total_rate_limited_requests),
        "total_requests": float(total_simulated_requests),
        "429_rate": float(total_rate_limited_requests) / max(1.0, float(total_simulated_requests)),
        "mean_concurrency": float(np.mean([step.concurrency for step in records])),
        "final_concurrency": float(records[-1].concurrency),
        "frac_near_true_capacity": float(fraction_steps_near_hidden_capacity),
    }


def print_pairwise_episode_metric_comparison(
    label_a: str,
    episode_metrics_a: Dict[str, float],
    label_b: str,
    episode_metrics_b: Dict[str, float],
) -> None:
    """
    Print a human-readable table comparing two episodes on the **same** metric keys.

    For each metric name present in ``episode_metrics_a`` (sorted lexicographically),
    prints one line with values for run *a*, run *b*, and
    ``delta = value_b - value_a`` (so positive delta means *b* is higher).

    Args:
        label_a:
            Short name for the first policy / experiment (e.g. ``"aimd"``).
        episode_metrics_a:
            Output of :func:`summarize_simulation_episode` for run *a*.
        label_b:
            Short name for the second policy / experiment (e.g. ``"model_guided"``).
        episode_metrics_b:
            Summary dict for run *b*; must contain the same keys as ``episode_metrics_a``
            for sensible deltas (if a key is missing, this function will raise).

    Returns:
        ``None`` (side effect: prints to stdout).

    Notes:
        * This is intentionally minimal—no matplotlib, no file output.
        * If you add new keys in :func:`summarize_simulation_episode`, they automatically
          show up here as long as both dicts stay aligned.
    """
    metric_names_sorted = sorted(episode_metrics_a.keys())
    print(f"\n=== {label_a} vs {label_b} ===")
    for metric_name in metric_names_sorted:
        value_a, value_b = episode_metrics_a[metric_name], episode_metrics_b[metric_name]
        print(
            f"{metric_name:26s}  {label_a}={value_a:.6g}  {label_b}={value_b:.6g}  "
            f"delta={value_b - value_a:+.6g}"
        )


def benchmark_classifier_predict_proba_nanoseconds(
    model: Any, feature_vector: np.ndarray, n_calls: int = 4000
) -> Dict[str, float]:
    """
    Estimate how many **nanoseconds** one call to the *wrapped* positive-class probability
    takes on average (H4 / inference cost).

    Uses :func:`features.classifier_predict_probability_batch_had_429` so timing includes
    the same reshaping and ``classes_`` lookup as production call sites. **Warmup**
    iterations run first to reduce one-off costs (import caches, CPU frequency ramp,
    branch predictor habits); the reported mean is over ``n_calls`` *timed* iterations
    only.

    Args:
        model:
            Fitted classifier (anything compatible with
            ``classifier_predict_probability_batch_had_429``).
        feature_vector:
            A single feature row, shape ``(n_features,)``, representative of what you pass
            during a real decision (e.g. a row from your training matrix).
        n_calls:
            Number of timed repetitions; larger values stabilize the mean but take longer.

    Returns:
        Dict with:

        - **mean_ns_per_call:** Average elapsed nanoseconds per call.
        - **mean_us_per_call:** Same quantity in microseconds (``ns / 1000``).

    Notes:
        * This is a **process-local** CPU micro-benchmark; other processes, power states,
          and Python GC can skew results.
        * Throughput in production also depends on batching, GIL contention, and whether
          you run inference in another thread—interpret these numbers as **order-of-magnitude**
          guidance, not a guaranteed SLA.
        * For very small ``n_calls``, variance is high; prefer thousands of iterations.
    """
    num_warmup_iterations = 40
    for _ in range(num_warmup_iterations):
        classifier_predict_probability_batch_had_429(model, feature_vector)
    start_time_ns = time.perf_counter_ns()
    for _ in range(n_calls):
        classifier_predict_probability_batch_had_429(model, feature_vector)
    end_time_ns = time.perf_counter_ns()
    nanoseconds_per_call = (end_time_ns - start_time_ns) / max(1, n_calls)
    return {
        "mean_ns_per_call": float(nanoseconds_per_call),
        "mean_us_per_call": float(nanoseconds_per_call) / 1000.0,
    }
