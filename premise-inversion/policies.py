"""
policies.py - run the fake "API client" for many time steps 

what this file does 
Imagine a loop . Each round : 

1. The client sends a **batch** of parallel requests . How parallel ? That is 
``concurrency`` ( how many at the same time )
2. The simulator (``sim_core.simulate_step``) answers: some requests succeed, some may
   return HTTP **429** (rate limited).
3. The client picks the **next** concurrency for the following round.


This loop runs ``num_steps`` times. We collect every round as a ``StepRecord`` in a list.
That list is what you train on in ``dataset.py`` or summarize in ``metrics.py``.

**"aimd"** — “Reactive” rule, no machine learning:

- If the batch had **no** 429: increase concurrency by a small step (see config).
- If the batch had **any** 429: shrink concurrency by multiplying by a factor < 1 (see
  config).

We still save the last few steps into a short memory, so experiments with and without ML
build **the same shape** of feature history.

*"model_guided"** — AIMD first, then optionally ask a model:

- We always compute what plain AIMD **would** do next.
- If AIMD wants to **raise** concurrency and the **current** batch was clean (no 429),
  we pause and ask a trained classifier: “How worried are we that the **next** batch
  would get a 429?”
- If the model’s answer is **low enough** (see threshold below), we **allow** the raise.
  If the answer is **too high**, we **do not** raise; we keep the old concurrency.
- If AIMD wants to **lower** concurrency (usually after 429s), we **always** allow that.
  The model never blocks “backing off”.

What the threshold means
--------------------------
``ml_threshold`` or ``config.ML_INCREASE_THRESHOLD`` is a number between 0 and 1.

Rule for **raising** only: we allow the raise when the following is true (``<=`` means
“less than or equal to”):

    predicted chance of 429  <=  threshold

The model must judge the danger as **at or below** your cutoff before you raise
parallelism. A **smaller** threshold means you only raise when the model looks **extra**
safe, so you attempt fewer risky jumps.

Memory we keep between rounds
------------------------------
* ``rolling_request_latencies_ms`` — many recent **single-request** latencies (numbers in
  ms). Used to compute rolling min / median / p95 inside ``features.py``. The training
  builder in ``dataset.py`` must update this deque the **same way**, or the model sees
  a different world at train vs run time.

* ``recent_steps_for_features`` — the last ``HISTORY_LEN`` full step summaries. That is
  what the encoder turns into one long vector of floats for sklearn.

Reproducible randomness
-----------------------
We call ``set_seed(config.RANDOM_SEED)`` once at the start so the **simulator’s** random
latencies match between runs if you did not change config.

"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import config
from features import encode_recent_step_history_as_feature_vector, classifier_predict_probability_batch_had_429
from metrics import summarize_simulation_episode
from sim_core import (
    StepRecord,
    aimd_adjust_concurrency,
    set_seed,
    simulate_step,
    update_rolling_p95,
)


def simulate_concurrency_policy_for_steps(
    policy: config.PolicyName,
    num_steps: int,
    *,
    model: Optional[Any] = None,
    ml_threshold: Optional[float] = None,
) -> Tuple[List[StepRecord], Dict[str, float]]:
    """
    Run the client+simulator loop ``num_steps`` times.

    Returns two things:

    1. A Python list of every ``StepRecord`` (one record per loop), oldest-first.
    2. A small dictionary of summary numbers (429 rate, average concurrency, …) from
       ``summarize_simulation_episode``.

    Inside each loop we always: simulate → update latency memory → save the record → pick
    the next concurrency (AIMD rule; sometimes the ML model can veto **raising** only).

    Parameters (beginner-friendly)
    ------------------------------
    policy
        Either ``"aimd"`` (rule only) or ``"model_guided"`` (rule + optional ML veto on
        increases). Any other string raises an error.

    num_steps
        How many rounds to run. Should be at least 1 if you later read ``records[-1]``.

    model
        Only for ``"model_guided"``. This should already be trained (``.fit`` done). If you
        forget to pass a model, you get ``ValueError``. For ``"aimd"``, you may leave it
        as ``None``.

    ml_threshold
        If you pass a number here, it **replaces** ``config.ML_INCREASE_THRESHOLD`` for
        this run only. It is the “how safe must the model say it is before we raise?”
        cutoff. ``None`` means “use the value from ``config.py``.”

    Advanced note for later
    -----------------------
    In ``"model_guided"`` we append the **current** step to ``recent_steps_for_features``
    *before* we turn history into numbers for the model. The training code in
    ``dataset.py`` uses a slightly different time rule on purpose. Read both files before
    you change either, or train/test may stop matching.

    This function never calls ``.fit``; training happens elsewhere.
    """
    # When the model says P(429) is *above* this number, we refuse to raise concurrency.
    max_probability_to_allow_increase = (
        config.ML_INCREASE_THRESHOLD if ml_threshold is None else float(ml_threshold)
    )

    # Fake latencies use numpy randomness; fix the seed so two runs with the same config match.
    set_seed(config.RANDOM_SEED)

    # Bucket 1: many recent single-request latency values (milliseconds).
    rolling_request_latencies_ms: Deque[float] = deque(maxlen=config.LAT_ROLLING_MAXLEN)
    # Bucket 2: the last few full steps (used to build one ML input row).
    recent_steps_for_features: Deque[StepRecord] = deque(maxlen=config.HISTORY_LEN)

    current_concurrency = 1  # start sending only one request at a time
    episode_step_records: List[StepRecord] = []  # grow one record per loop

    for step_index in range(num_steps):
        # Ask the simulator: “What happens if we send `current_concurrency` parallel calls?”
        current_step_record = simulate_step(step_index, current_concurrency)

        # Fill in `latency_p95_recent` and push each sample latency into the rolling bucket.
        current_step_record.latency_p95_recent = update_rolling_p95(
            rolling_request_latencies_ms, current_step_record.latency_ms_list
        )
        episode_step_records.append(current_step_record)

        if policy == "aimd":
            # Plain rule: go up a little after success, down a lot after any 429.
            current_concurrency = aimd_adjust_concurrency(
                current_concurrency,
                current_step_record.had_any_429,
                config.AIMD_ADD_ON_SUCCESS,
                config.AIMD_MULT_ON_429,
            )
            # Remember this step so the short history looks like the guided policy’s history.
            recent_steps_for_features.append(current_step_record)

        elif policy == "model_guided":
            if model is None:
                raise ValueError("model_guided needs a fitted model")

            # History always includes the step we *just* finished (we are about to decide the next level).
            recent_steps_for_features.append(current_step_record)

            # First ask: what would dumb AIMD do?
            aimd_proposed_concurrency = aimd_adjust_concurrency(
                current_concurrency,
                current_step_record.had_any_429,
                config.AIMD_ADD_ON_SUCCESS,
                config.AIMD_MULT_ON_429,
            )

            # Special case: batch was clean *and* AIMD wants to go **up**. That is the only time ML may say “stop.”
            if current_step_record.had_any_429 == 0 and aimd_proposed_concurrency > current_concurrency:
                history_feature_vector = encode_recent_step_history_as_feature_vector(
                    recent_steps_for_features, rolling_request_latencies_ms
                )
                predicted_batch_429_probability = classifier_predict_probability_batch_had_429(
                    model, history_feature_vector
                )
                if predicted_batch_429_probability <= max_probability_to_allow_increase:
                    current_concurrency = aimd_proposed_concurrency
                # If we land here with no change: model said risk is too high, stay flat.
            else:
                # Either we had 429s, or AIMD is not trying to increase — always follow AIMD.
                current_concurrency = aimd_proposed_concurrency
        else:
            raise ValueError(policy)

    return episode_step_records, summarize_simulation_episode(episode_step_records)


