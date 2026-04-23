# Heuristics Practitioners Learn the Hard Way

## 1) Nyquist heuristic for window size

Set `W` to at least **2x** the period of the dominant seasonality.  
If unsure, plot the spectrum (via FFT) and read the dominant peaks.

## 2) "Newspaper test" for leakage

For every feature, ask:

> If I were a trader with only a newspaper from time `t`, could I compute this feature?

If the answer is **no**, the feature is leaking.

Feature leakage means the feature contains information that would not have been available at prediction time, making evaluation unrealistically optimistic.

## 3) Correlation sanity check

After splitting, check that Pearson correlation between adjacent train/test series is not suspiciously high.  
In a correct split, correlation should reflect true autocorrelation, not inflated overlap.

## 4) Stride is a speed-vs-accuracy trade-off

- `s = 1`: most training data, but highly correlated examples.
- `s = W`: more independent examples, but less data.
- For deep learning models, `s = 1` is often better.
- For statistical models, use `s >= H`, where `H` is the forecast horizon.

