"""
features.py

Turns recent simulator *history* into a single numeric vector that sklearn (or any
tabular model) can consume, and safely reads `predict_proba` output for the positive
class “this HTTP batch included a 429.”

**Time discipline:** whoever calls ``encode_recent_step_history_as_feature_vector`` must do
so *before* folding the current step’s ``StepRecord`` into history if the label is “did
*this* step 429?”—see ``dataset.py``. Otherwise you leak the future and the classifier
looks unrealistically good.

This module does **not** import sklearn; it only assumes the model exposes
``predict_proba`` and ``classes_`` like a typical ``sklearn`` classifier.
"""

from __future__ import annotations

from typing import Any, Deque, List

import numpy as np

import config
from sim_core import StepRecord


def classifier_predict_probability_batch_had_429(model: Any, feature_vector: np.ndarray) -> float:
    """
    Return the model’s estimated probability that label **1** applies, where label **1**
    means *at least one request in the batch received HTTP 429*.

    **Why not just ``predict_proba(...)[:, 1]``?**
    Scikit-learn orders columns of ``predict_proba`` to match ``model.classes_``, which
    is not guaranteed to be ``[0, 1]`` (e.g. if the training set temporarily had only one
    class, or classes are sorted differently). Indexing by ``classes_.index(1)`` picks the
    column that actually corresponds to the positive class.

    Args:
        model:
            Any fitted classifier with ``predict_proba`` and ``classes_`` (e.g.
            ``GradientBoostingClassifier``, ``LogisticRegression``).
        feature_vector:
            One *row* of features as a 1-D array, shape ``(n_features,)``. It will be
            reshaped to ``(1, n_features)`` because sklearn expects a 2-D design matrix.

    Returns:
        A float in ``[0, 1]`` (subject to floating error): *P*(batch contained a 429 |
        ``feature_vector``), according to the model.

    Notes:
        - This is a thin, defensive wrapper; all timing / batching policy lives in
          ``policies.py`` and friends.
        - For **calibrated** probabilities you would add ``CalibratedClassifierCV`` or
          similar upstream; this helper does not change the model.
    """
    probability_per_class = model.predict_proba(feature_vector.reshape(1, -1))[0]
    fitted_class_order = list(getattr(model, "classes_", [0, 1]))
    positive_class_column_index = int(fitted_class_order.index(1))
    return float(probability_per_class[positive_class_column_index])


def encode_recent_step_history_as_feature_vector(
    step_hist: Deque[StepRecord],
    rolling_latencies: Deque[float],
) -> np.ndarray:
    """
    Flatten the last ``HISTORY_LEN`` steps plus a small summary of the rolling latency
    deque into **one** float vector for a classifier.

    **Per-step block (repeated up to ``HISTORY_LEN`` times, oldest → newest within the
    slice):**

    1. ``concurrency`` — client parallelism that step.
    2. ``had_any_429`` — 0/1 flag for that step’s batch.
    3. ``latency_p95_recent`` — tail latency statistic stored on that ``StepRecord`` *as of
       that step* (not recomputed here from raw request samples).

    **Tail block (three floats):** min, median (50th percentile), and 95th percentile of
    *all* individual request latencies currently inside ``rolling_latencies``. That deque
    is filled by ``sim_core.update_rolling_p95`` and tracks recent raw samples across
    steps, so this block can capture global queueing stress beyond a single step’s summary.

    **Padding:** If fewer than ``HISTORY_LEN`` steps exist (early in an episode), the
    missing prefix is filled with zeros so each row has fixed length ``3 * HISTORY_LEN +
    3``. Zeros are a crude “no history” signal; a richer design might use masks or learned
    embeddings.

    Args:
        step_hist:
            Newest-at-right deque of ``StepRecord`` (max length typically ``HISTORY_LEN``);
            only the last ``HISTORY_LEN`` entries are read if it is longer.
        rolling_latencies:
            Sliding window of per-request latency samples (milliseconds), same object the
            simulator updates when stepping.

    Returns:
        ``np.ndarray`` of dtype ``float64``, shape ``(3 * HISTORY_LEN + 3,)``.

    Notes:
        * NaN values in ``latency_p95_recent`` are treated as 0.0 so tree/linear models
          never see NaNs from uninitialized records.
        * The **layout** must stay in sync with ``dataset.py`` / ``policies.py``:
          changing the order or count of features without retraining breaks the model.
    """
    flat_feature_values: List[float] = []
    recent_steps_chronological = list(step_hist)

    if len(recent_steps_chronological) < config.HISTORY_LEN:
        for _ in range(config.HISTORY_LEN - len(recent_steps_chronological)):
            flat_feature_values.extend([0.0, 0.0, 0.0])

    for past_step in recent_steps_chronological[-config.HISTORY_LEN :]:
        step_tail_latency_p95_ms = past_step.latency_p95_recent
        if not (step_tail_latency_p95_ms == step_tail_latency_p95_ms):  # NaN check without math.isnan
            step_tail_latency_p95_ms = 0.0
        flat_feature_values.extend(
            [
                float(past_step.concurrency),
                float(past_step.had_any_429),
                float(step_tail_latency_p95_ms),
            ]
        )

    per_step_features_flat = np.asarray(flat_feature_values, dtype=np.float64)

    if len(rolling_latencies) > 0:
        rolling_request_latencies_ms = np.asarray(rolling_latencies, dtype=np.float64)
        rolling_window_min_median_p95_ms = [
            float(np.min(rolling_request_latencies_ms)),
            float(np.percentile(rolling_request_latencies_ms, 50)),
            float(np.percentile(rolling_request_latencies_ms, 95)),
        ]
    else:
        rolling_window_min_median_p95_ms = [0.0, 0.0, 0.0]

    return np.concatenate(
        [per_step_features_flat, np.asarray(rolling_window_min_median_p95_ms, dtype=np.float64)],
        axis=0,
    )
