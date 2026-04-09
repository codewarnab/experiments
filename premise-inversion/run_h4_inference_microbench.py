"""
run_h4_inference_microbench.py — H4: micro-benchmark predict path cost
"""
import config
from dataset import build_labeled_training_arrays_from_step_records
from metrics import benchmark_classifier_predict_proba_nanoseconds
from policies import simulate_concurrency_policy_for_steps
from sim_core import set_seed
from training import fit_gradient_boosting_rate_limit_classifier


def main() -> None:
    set_seed(config.RANDOM_SEED)
    aimd_episode_records, _ = simulate_concurrency_policy_for_steps("aimd", config.NUM_STEPS)
    feature_matrix, labels_batch_had_429 = build_labeled_training_arrays_from_step_records(
        aimd_episode_records
    )
    rate_limit_classifier = fit_gradient_boosting_rate_limit_classifier(
        feature_matrix, labels_batch_had_429
    )
    mid_episode_feature_row = feature_matrix[len(feature_matrix) // 2]
    print(
        benchmark_classifier_predict_proba_nanoseconds(
            rate_limit_classifier, mid_episode_feature_row, n_calls=8000
        )
    )


if __name__ == "__main__":
    main()
