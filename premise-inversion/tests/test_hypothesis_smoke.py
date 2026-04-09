"""Smoke tests for H1–H5 pipeline pieces (short episodes via conftest)."""
import pytest

import config
from dataset import build_labeled_training_arrays_from_step_records
from metrics import benchmark_classifier_predict_proba_nanoseconds, summarize_simulation_episode
from policies import simulate_concurrency_policy_for_steps
from policy_ewma import simulate_aimd_episode_with_ewma_latency_guard
from sim_core import simulate_step
from training import fit_gradient_boosting_rate_limit_classifier, fit_logistic_rate_limit_classifier


def test_simulate_step_respects_capacity() -> None:
    r = simulate_step(0, config.TRUE_CAPACITY + 5)
    assert r.num_ok == config.TRUE_CAPACITY
    assert r.num_429 == 5
    assert r.had_any_429 == 1


def test_dataset_shape_matches_episode_minus_one() -> None:
    records, _ = simulate_concurrency_policy_for_steps("aimd", config.NUM_STEPS)
    X, y = build_labeled_training_arrays_from_step_records(records)
    assert X.shape[0] == len(records) - 1
    assert y.shape[0] == len(records) - 1
    assert X.shape[1] == 3 * config.HISTORY_LEN + 3


def test_model_guided_requires_model() -> None:
    with pytest.raises(ValueError, match="model_guided needs"):
        simulate_concurrency_policy_for_steps("model_guided", 10, model=None)


def test_h1_aimd_train_model_guided_roundtrip() -> None:
    aimd_records, aimd_m = simulate_concurrency_policy_for_steps("aimd", config.NUM_STEPS)
    X, y = build_labeled_training_arrays_from_step_records(aimd_records)
    clf = fit_gradient_boosting_rate_limit_classifier(X, y)
    _, guided_m = simulate_concurrency_policy_for_steps("model_guided", config.NUM_STEPS, model=clf)
    for k in aimd_m:
        assert k in guided_m
    assert 0.0 <= guided_m["429_rate"] <= 1.0


def test_h3_logistic_and_gbdt_run() -> None:
    aimd_records, _ = simulate_concurrency_policy_for_steps("aimd", config.NUM_STEPS)
    X, y = build_labeled_training_arrays_from_step_records(aimd_records)
    gb = fit_gradient_boosting_rate_limit_classifier(X, y)
    lr = fit_logistic_rate_limit_classifier(X, y)
    assert gb.predict_proba(X[:1]).shape == (1, 2)
    assert lr.predict_proba(X[:1]).shape == (1, 2)


def test_h4_benchmark_positive() -> None:
    aimd_records, _ = simulate_concurrency_policy_for_steps("aimd", config.NUM_STEPS)
    X, y = build_labeled_training_arrays_from_step_records(aimd_records)
    clf = fit_gradient_boosting_rate_limit_classifier(X, y)
    row = X[min(5, len(X) - 1)]
    stats = benchmark_classifier_predict_proba_nanoseconds(clf, row, n_calls=50)
    assert stats["mean_ns_per_call"] > 0


def test_h5_ewma_episode_summarizes() -> None:
    records, m = simulate_aimd_episode_with_ewma_latency_guard(config.NUM_STEPS)
    assert len(records) == config.NUM_STEPS
    assert m == summarize_simulation_episode(records)


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        simulate_concurrency_policy_for_steps("typo", 5)  # type: ignore[arg-type]
