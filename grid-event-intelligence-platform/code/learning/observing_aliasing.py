# This file shows "aliasing" in a very simple way.
# Aliasing means: when we sample too slowly, a fast wave can look like a slower wave.

# NumPy helps us do math with arrays (lists of numbers).
import numpy as np

# Matplotlib helps us draw plots/graphs.
import matplotlib.pyplot as plt

# Real/original signal frequency (7 cycles per second = 7 Hz)
f_true = 7.0

# Make many time points from 0 to 1 second.
# Think of this as a "smooth timeline" so the curve looks continuous.
t_continuous = np.linspace(0, 1, 10000)

# Create the original sine wave on the smooth timeline.
# Formula for sine wave: sin(2*pi*frequency*time)
signal_true = np.sin(2 * np.pi * f_true * t_continuous)

# Sampling frequency = 10 Hz (10 samples per second)
f_s = 10.0

# Take sample times every (1/f_s) seconds.
# np.arange(start, stop, step) gives values from start to just before stop.
t_sampled = np.arange(0, 1, 1.0 / f_s)

# Evaluate the same 7 Hz wave only at sampled times.
signal_sampled = np.sin(2 * np.pi * f_true * t_sampled)

# Nyquist frequency is f_s/2 = 5 Hz.
# Since 7 Hz > 5 Hz, the sampled data cannot represent it correctly.
# It appears as a lower "fake" frequency (alias).
# This line computes that alias frequency.
f_alias = abs(f_true - round(f_true / f_s) * f_s)

# Build a smooth wave for the alias frequency, so we can compare visually.
signal_alias = np.sin(2 * np.pi * f_alias * t_continuous)

# -----------------------
# Print simple calculation logs before plot opens
# -----------------------
nyquist = f_s / 2.0
sample_step = 1.0 / f_s
k = round(f_true / f_s)
sample_count = len(t_sampled)

print("\n=== Aliasing Calculation Log ===")
print(f"True frequency (f_true): {f_true:.2f} Hz")
print(f"Sampling frequency (f_s): {f_s:.2f} Hz")
print(f"Nyquist frequency (f_s/2): {nyquist:.2f} Hz")
print(f"Time between samples (1/f_s): {sample_step:.4f} s")
print(f"Number of samples in 1 second: {sample_count}")
print(f"k = round(f_true / f_s) = round({f_true:.2f}/{f_s:.2f}) = {k}")
print(f"Alias formula: f_alias = |f_true - k*f_s|")
print(f"f_alias = |{f_true:.2f} - {k}*{f_s:.2f}| = {f_alias:.2f} Hz")
print(
    "Meaning: your 7 Hz signal is sampled so slowly that it appears as "
    f"a {f_alias:.2f} Hz wave in sampled data."
)
print("Now opening plot window...\n")

# Create one figure window.
plt.figure(figsize=(10, 5))

# Plot the original (real) wave as a solid line.
plt.plot(t_continuous, signal_true, label=f"True signal ({f_true:.1f} Hz)", linewidth=2)

# Plot the alias wave as a dashed line.
plt.plot(
    t_continuous,
    signal_alias,
    "--",
    label=f"Aliased appearance ({f_alias:.1f} Hz)",
    linewidth=2,
)

# Draw sampled points and vertical stems to show discrete sampling.
plt.stem(
    t_sampled,
    signal_sampled,
    linefmt="C3-",
    markerfmt="C3o",
    basefmt="k-",
    label=f"Samples at {f_s:.1f} Hz",
)

# Add plot labels and styling.
plt.title("Aliasing Example: 7 Hz signal sampled at 10 Hz")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.legend()

# Improve spacing so labels do not overlap.
plt.tight_layout()

# Display the plot on screen.
plt.show()