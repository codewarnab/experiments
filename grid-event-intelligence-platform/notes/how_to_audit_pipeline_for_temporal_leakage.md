# What Temporal Leakage Means

Temporal leakage happens when a model is trained with information that would not have been available at prediction time.  
In a grid event pipeline, this usually means features for time `t` accidentally include data from `t+1` or later (for example, future outages, future weather aggregates, or labels that were joined too early).  
This makes offline validation look unrealistically good and then fail in production, because the model learned from the future instead of only the past.


