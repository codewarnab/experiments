"""
sim_core.py 

Turns a "concurrency level" into a batch of fake http results : some ok , some 429 
when you go over the hidden limit . Also holds the small AIMD rule (raise after 
good batch cut after 429 ).

(raise after good batch , cut after 429 ).

There is " no machine learning " this file only simpel match and randome latency >
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np 
import config 

def  set_seed(seed:int ) -> None : 
    """ Pick a fixed random sequence so reruns match when you use the same seed ."""
    np.random.seed(seed) 


@dataclass
class StepRecord:
    step_index: int 
    concurrency: int 
    num_ok : int 
    num_429 : int 
    had_any_429: int # 1 if any requet in this batch was 429 
    latency_ms_list: List[float] 
    latency_p95_recent : float 


def sample_latencies(num_ok: int, num_429: int, concurrency: int) -> Tuple[List[float], List[float]]:
    """Make fake request latencies. If LAT_CONGESTION_GAIN_MS > 0, OK latencies creep up before 429s (H2)."""
    cap = float(config.TRUE_CAPACITY)  # hidden limit; used only inside the simulator
    # Roughly 0 at low concurrency, approaches ~1 as concurrency passes the cap (precursor signal for H1/H2).
    congestion_proxy = max(0.0, (concurrency - 1) / max(1.0, cap))
    # *10 scales the gain so the bump is visible on top of LAT_BASE_MS with typical config values.
    ok_mean = config.LAT_BASE_MS + config.LAT_CONGESTION_GAIN_MS * congestion_proxy * 10.0
    ok_latencies = np.random.normal(loc=ok_mean, scale=config.LAT_NOISE_STD_MS, size=max(0, num_ok)).tolist()
    bad_mean = max(1.0, 0.35 * config.LAT_BASE_MS)  # 429s somewhat faster than healthy OKs (tunable theater)
    bad_latencies = np.random.normal(
        loc=bad_mean, scale=0.6 * config.LAT_NOISE_STD_MS, size=max(0, num_429)
    ).tolist()
    return [float(x) for x in ok_latencies], [float(x) for x in bad_latencies]


def simulate_step(step_index: int, concurrency: int) -> StepRecord:
    """Run one batch: up to TRUE_CAPACITY requests succeed; extra requests get 429."""
    concurrency = max(1, int(concurrency))
    # Deterministic split: first TRUE_CAPACITY “slots” succeed; anything beyond is rate-limited.
    num_ok = min(concurrency, config.TRUE_CAPACITY)
    num_429 = concurrency - num_ok
    ok_lats, bad_lats = sample_latencies(num_ok, num_429, concurrency=concurrency)
    all_lats = ok_lats + bad_lats  # feature code can use min/p95 over this list per step
    return StepRecord(
        step_index=step_index,
        concurrency=concurrency,
        num_ok=num_ok,
        num_429=num_429,
        had_any_429=1 if num_429 > 0 else 0,
        latency_ms_list=all_lats,
        latency_p95_recent=float(np.nan),  # not computed here—caller updates rolling deque then sets p95
    )


def update_rolling_p95(rolling_latencies: Deque[float], new_latencies: List[float]) -> float:
    """Add new latencies to a sliding window and return the 95th percentile (recent tail latency)."""
    for x in new_latencies:
        rolling_latencies.append(float(x))  # maxlen on the deque drops oldest samples automatically
    if len(rolling_latencies) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(rolling_latencies, dtype=np.float64), 95))


def aimd_adjust_concurrency(concurrency: int, had_any_429: int, add: int, mult: float) -> int:
    """AIMD rule: increase by `add` if no 429; multiply by `mult` (and floor) if there was a 429."""
    c = max(1, int(concurrency))
    if had_any_429:
        # Multiplicative decrease: large cutback after the server signals overload (429).
        return max(1, int(np.floor(c * mult)))
    # Additive increase: grow slowly while batches are clean (TCP-style probe upward).
    return c + int(add)