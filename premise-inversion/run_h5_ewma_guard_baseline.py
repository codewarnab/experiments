"""
run_h5_ewma_guard_baseline.py — H5: EWMA guard vs plain AIMD
"""
import config
from metrics import print_pairwise_episode_metric_comparison
from policies import simulate_concurrency_policy_for_steps
from policy_ewma import simulate_aimd_episode_with_ewma_latency_guard
from sim_core import set_seed


def main() -> None:
    set_seed(config.RANDOM_SEED)
    _, plain_aimd_episode_summary_metrics = simulate_concurrency_policy_for_steps(
        "aimd", config.NUM_STEPS
    )
    _, ewma_guard_episode_summary_metrics = simulate_aimd_episode_with_ewma_latency_guard(
        config.NUM_STEPS
    )
    print_pairwise_episode_metric_comparison(
        "aimd",
        plain_aimd_episode_summary_metrics,
        "aimd_ewma_guard",
        ewma_guard_episode_summary_metrics,
    )


if __name__ == "__main__":
    main()
