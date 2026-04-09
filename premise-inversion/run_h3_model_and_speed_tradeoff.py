"""
run_h3_model_and_speed_tradeoff.py — H3: model prestige vs inference speed
"""
import config
from dataset import build_labeled_training_arrays_from_step_records
from metrics import (
    benchmark_classifier_predict_proba_nanoseconds,
    print_pairwise_episode_metric_comparison,
)
from policies import simulate_concurrency_policy_for_steps
from sim_core import set_seed
from training import fit_gradient_boosting_rate_limit_classifier, fit_logistic_rate_limit_classifier


def main() -> None:
    set_seed(config.RANDOM_SEED)
    aimd_episode_records, _ = simulate_concurrency_policy_for_steps("aimd", config.NUM_STEPS)
    feature_matrix, labels_batch_had_429 = build_labeled_training_arrays_from_step_records(
        aimd_episode_records
    )
    sample_row_index = min(len(feature_matrix) - 1, 200)
    single_feature_row_for_timing = feature_matrix[sample_row_index]

    gradient_boosting_model = fit_gradient_boosting_rate_limit_classifier(
        feature_matrix, labels_batch_had_429
    )
    logistic_model = fit_logistic_rate_limit_classifier(feature_matrix, labels_batch_had_429)
    _, gbdt_episode_summary_metrics = simulate_concurrency_policy_for_steps(
        "model_guided", config.NUM_STEPS, model=gradient_boosting_model
    )
    _, logistic_episode_summary_metrics = simulate_concurrency_policy_for_steps(
        "model_guided", config.NUM_STEPS, model=logistic_model
    )
    print_pairwise_episode_metric_comparison(
        "gbdt", gbdt_episode_summary_metrics, "logistic", logistic_episode_summary_metrics
    )

    print("\nH3 inference (mean per probability call):")
    print(
        "  gbdt:    ",
        benchmark_classifier_predict_proba_nanoseconds(
            gradient_boosting_model, single_feature_row_for_timing
        ),
    )
    print(
        "  logistic:",
        benchmark_classifier_predict_proba_nanoseconds(logistic_model, single_feature_row_for_timing),
    )


if __name__ == "__main__":
    main()
