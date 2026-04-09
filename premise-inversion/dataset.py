"""
dataset.py — turn a finished simulator run into supervised-learning data

Replay step records with the same rolling deques as ``policies.py`` so training
features match deployment-time encodings.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import numpy as np

import config
from features import encode_recent_step_history_as_feature_vector
from sim_core import StepRecord, update_rolling_p95


def build_labeled_training_arrays_from_step_records(records: List[StepRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build ``X`` and binary ``y`` from one episode. Row i predicts ``had_any_429`` at
    step i using history through step i - 1 only (no peeking at step i latencies).
    """
    feature_rows: List[np.ndarray] = []
    labels_batch_had_429: List[int] = []
    rolling_request_latencies_ms: Deque[float] = deque(maxlen=config.LAT_ROLLING_MAXLEN)
    recent_steps_for_features: Deque[StepRecord] = deque(maxlen=config.HISTORY_LEN)

    for step_index in range(len(records)):
        record_at_time_t = records[step_index]

        if step_index == 0:
            record_at_time_t.latency_p95_recent = update_rolling_p95(
                rolling_request_latencies_ms, record_at_time_t.latency_ms_list
            )
            recent_steps_for_features.append(record_at_time_t)
            continue

        feature_rows.append(
            encode_recent_step_history_as_feature_vector(
                recent_steps_for_features, rolling_request_latencies_ms
            )
        )
        labels_batch_had_429.append(int(record_at_time_t.had_any_429))

        record_at_time_t.latency_p95_recent = update_rolling_p95(
            rolling_request_latencies_ms, record_at_time_t.latency_ms_list
        )
        recent_steps_for_features.append(record_at_time_t)

    return np.vstack(feature_rows), np.asarray(labels_batch_had_429, dtype=np.int64)
