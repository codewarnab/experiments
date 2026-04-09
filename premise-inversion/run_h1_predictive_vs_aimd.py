"""
run_h1_predictive_vs_aimd.py — H1: ML-gated AIMD vs plain AIMD
"""
from sklearn.metrics import classification_report

import config
from dataset import build_labeled_training_arrays_from_step_records
from metrics import print_pairwise_episode_metric_comparison
from policies import simulate_concurrency_policy_for_steps
from sim_core import set_seed
from training import fit_gradient_boosting_rate_limit_classifier


def main() -> None:
    set_seed(config.RANDOM_SEED)
    aimd_episode_records, aimd_episode_summary_metrics = simulate_concurrency_policy_for_steps(
        "aimd", config.NUM_STEPS
    )
    feature_matrix, labels_batch_had_429 = build_labeled_training_arrays_from_step_records(
        aimd_episode_records
    )
    print(f"H1  X={feature_matrix.shape}  y rate(429 in batch)={labels_batch_had_429.mean():.4f}")
    rate_limit_classifier = fit_gradient_boosting_rate_limit_classifier(
        feature_matrix, labels_batch_had_429
    )
    print("\nNote: scores below are in-sample (rosy). For a stricter check, run H2.\n")
    print(
        classification_report(
            labels_batch_had_429, rate_limit_classifier.predict(feature_matrix), digits=4
        )
    )
    _, model_guided_episode_summary_metrics = simulate_concurrency_policy_for_steps(
        "model_guided", config.NUM_STEPS, model=rate_limit_classifier
    )
    print_pairwise_episode_metric_comparison(
        "aimd", aimd_episode_summary_metrics, "model_guided", model_guided_episode_summary_metrics
    )


if __name__ == "__main__":
    main()
