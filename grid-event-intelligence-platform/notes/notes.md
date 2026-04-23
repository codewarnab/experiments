# What Nyquist Means When You Build Models

## 1) Your sampling rate determines what you can learn

If you sample sensor data at `10 Hz`, you can never detect events that happen faster than `5 Hz`, no matter how powerful your model is.

## 2) Downsampling destroys information

If you average your data into hourly buckets because it is easier, you permanently throw away all sub-hourly patterns. Make this decision consciously.

## 3) Irregular sampling is dangerous

Nyquist assumes uniform sampling. Missing values, dropped packets, or irregular polling intervals create frequency ambiguities that can appear as spurious patterns.

## 4) Upsampling does not add information

Interpolating `10 Hz` data to `100 Hz` and feeding it to a model does not let the model see frequencies between `5 Hz` and `50 Hz`. You are only creating synthetic points derived from the original ones.

---

# The Window Size Hyperparameter

`W` is one of the most consequential hyperparameters in time-series modeling. It controls how far back the model is allowed to look.

- **Too small:** The model cannot see periodic patterns that span multiple cycles. If you are modeling weekly electricity demand but your window is only 12 hours wide, the model has no idea what day of the week it is.
- **Too large:** You reduce the number of training examples (since `n - W` decreases), include potentially uninformative or misleading old context, and increase model complexity and training time.

---

# Relating Window Size to Nyquist

There is a direct connection between window size and what frequencies the model can detect.

A model that sees a window of `W` time steps (sampled at rate `fs`) can detect periodic patterns with minimum detectable frequency:

```text
fmin = fs / W
```

To detect a pattern with period `P` (in samples), you need at least:

```text
W >= P
```

In other words: `window size >= longest period you care about`.

---

# Stride in Time-Series ML

`stride` is how many time steps you move the sliding window forward to create the next training example.

If `window size = 5` and data is:

```text
[1,2,3,4,5,6,7,8,9,10]
```

- `stride = 1` -> `[1,2,3,4,5]`, `[2,3,4,5,6]`, `[3,4,5,6,7]`, ...
- `stride = 2` -> `[1,2,3,4,5]`, `[3,4,5,6,7]`, `[5,6,7,8,9]`, ...

Trade-offs:

- **Smaller stride (e.g., 1):** more overlap, more training samples, slower training.
- **Larger stride:** less overlap, fewer samples, faster training, but potentially less detail.

Related terms (often confused):

- **Window size (`W`)** = how far back the model looks.
- **Stride** = how far the window moves each step.
- **Forecast horizon (`H`)** = how far ahead you predict.