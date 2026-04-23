import numpy as np 
import matplotlib.pyplot as plt

def sliding_windows(series : np.ndarray , window_size :int , horizon : int = 1 , stride : int  =1 ) : 
    """ 
    Create sliding window examples from 1 - D series 
    parameters 
    -------------
    series : 1 - D array of shape (N,)
    window_size : number past steps in each input (w) 
    horizon : number of futyre steps to predic (H)
    stride : step size between consecutive windows

    Returns 
    ---------------
    X : shape ( n_samples , window_size ) -- inputs 
    y: shape (n _samples , horizon )   -- targets  
    """
    N=len(series) 
    #Total length consumed by one (input , target ) pair 
    total = window_size + horizon 

    indices = range(0,N - total + 1 , stride )
    X = np.array([series[i:i+window_size] for i in indices ])  # Build input matrix from sliding windows of length `window_size`.
    y = np.array([series[i + window_size : i + total ] for i in indices ])

    return X , y 


#Example 

np.random.seed(42) 
#simulate a noisy sine wave (e.g hourly data for 500 hours )
t = np.arange(500)
ts = np.sin(2*np.pi*t/24) + 0.3 * np.random.randn(500) 

W = 48 # look back 48 hours 
H = 24 # predict next 24 hours 
S = 1 #slide one step at a time 

X , y = sliding_windows(ts,window_size=W,horizon=H,stride=S)
print(F"Dataset shape: {X.shape}")
print(F"Input shape: {X[0].shape}")
print(F"Target shape: {y[0].shape}")

# Plot full series + multiple (input, target) training examples
sample_indices = [0, 24, 48]
fig, axes = plt.subplots(len(sample_indices), 1, figsize=(11, 9), sharex=True, sharey=True)

for ax, sample_idx in zip(axes, sample_indices):
    start = sample_idx * S
    x_input_idx = np.arange(start, start + W)
    x_target_idx = np.arange(start + W, start + W + H)

    ax.plot(t, ts, label="Full time series", color="steelblue", alpha=0.45)
    ax.plot(x_input_idx, X[sample_idx], label=f"Input window (W={W})", color="orange", linewidth=2)
    ax.plot(x_target_idx, y[sample_idx], label=f"Target horizon (H={H})", color="green", linewidth=2)
    ax.set_title(f"Sliding-window example: sample {sample_idx}")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Time step")
plt.tight_layout()
plt.show()

# What we learned for time-series prediction:
# 1) Sliding windows convert one long sequence into many supervised (X, y) examples.
# 2) W (window size) controls how much past context the model sees.
# 3) H (horizon) controls how far ahead we predict; larger H is usually harder.
# 4) S (stride) controls overlap between samples; S=1 gives many highly similar windows.
# 5) Correct alignment is critical: target starts AFTER input ends (prevents leakage).
# 6) Visualizing multiple samples (0, 24, 48) confirms the window shifts through time.


