"""
run_h2_precursor_ablation.py — H2: precursor signal ablation (LAT_CONGESTION_GAIN_MS)
"""
import config
from dataset import build_labeled_training_arrays_from_step_records
from metrics import print_pairwise_episode_metric_comparison
from policies import simulate_concurrency_policy_for_steps
from sim_core import set_seed
from training import evaluate_classifier_with_chronological_split, fit_gradient_boosting_rate_limit_classifier


def run_at_gain(gain_ms: float) -> None:
    previous_congestion_gain_ms = config.LAT_CONGESTION_GAIN_MS
    config.LAT_CONGESTION_GAIN_MS = gain_ms
    try:
        set_seed(config.RANDOM_SEED)
        print(f"\n######## LAT_CONGESTION_GAIN_MS = {gain_ms} ########")
        aimd_episode_records, aimd_episode_summary_metrics = simulate_concurrency_policy_for_steps(
            "aimd", config.NUM_STEPS
        )
        feature_matrix, labels_batch_had_429 = build_labeled_training_arrays_from_step_records(
            aimd_episode_records
        )
        print(f"  X={feature_matrix.shape}  positive_rate={labels_batch_had_429.mean():.4f}")
        evaluate_classifier_with_chronological_split(feature_matrix, labels_batch_had_429, train_frac=0.7)
        rate_limit_classifier = fit_gradient_boosting_rate_limit_classifier(
            feature_matrix, labels_batch_had_429
        )
        _, model_guided_episode_summary_metrics = simulate_concurrency_policy_for_steps(
            "model_guided", config.NUM_STEPS, model=rate_limit_classifier
        )
        print_pairwise_episode_metric_comparison(
            "aimd", aimd_episode_summary_metrics, "model_guided", model_guided_episode_summary_metrics
        )
    finally:
        config.LAT_CONGESTION_GAIN_MS = previous_congestion_gain_ms


def main() -> None:
    run_at_gain(0.0)
    run_at_gain(0.55)


if __name__ == "__main__":
    main()
