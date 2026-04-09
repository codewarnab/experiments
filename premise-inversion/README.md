# Premise inversion experiments

Baseline = **plain AIMD** (raise on success, cut on 429). Gains below are from one full run with default `config.py`.

| Hypothesis | Idea | Gain vs baseline |
|------------|------|------------------|
| **H1** | Use a small ML model to **allow increases only when it looks safe** | Fewer 429s (554 → 498), higher concurrency, ends at cap (20 vs 14) |
| **H2** | Learning needs **latency creeping before 429**; turn that off vs on | **No difference** in our simulator—trajectory doesn’t use latency, so baseline + model numbers matched |
| **H3** | Fancy model (GBDT) vs **simple logistic** | **Same** behavior; logistic predicts ~2.5× **faster** per call |
| **H4** | How **expensive** is one model prediction? | ~100 µs per GBDT call (order of magnitude for hot path) |
| **H5** | **EWMA latency rule** instead of ML | Slightly fewer 429s (554 → 541), small bump in concurrency—not as strong as H1 |

More detail: `h1-premise-inversion-experiment.md` (when present in this folder).

From this directory, run: `run_h1_predictive_vs_aimd.py` through `run_h5_ewma_guard_baseline.py`. Tests: `pytest` (or `python -m pytest`).
