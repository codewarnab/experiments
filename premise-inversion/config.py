"""
config.py 

all settings live here server capacity , run length , AIMD behaviour , fake latency 
and how strict the ML "Shoudl we raise concurrent?" gate is 

when you test a hypetheisis change **one** knob at a 
so you know what casued the diffece .
"""
from typing import Literal

RANDOM_SEED = 7 
TRUE_CAPACITY = 20 
NUM_STEPS = 5000 

AIMD_ADD_ON_SUCCESS = 1 
AIMD_MULT_ON_429 = 0.65

LAT_BASE_MS = 8.0
LAT_NOISE_STD_MS = 2.0
# H2 ablation: 0.0 removes “latency rises before 429” structure; 0.55 keeps precursors.
LAT_CONGESTION_GAIN_MS = 0.55

LAT_ROLLING_MAXLEN = 200
HISTORY_LEN = 10

# H1: higher = more cautious about increasing (tune and record tradeoffs).
ML_INCREASE_THRESHOLD = 0.35

PolicyName = Literal["aimd", "model_guided"]