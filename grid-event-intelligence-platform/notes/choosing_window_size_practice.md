1. start domain driven : what is the longest cycle in your data ?
Daily , Weekly , Monthly , Yearly , etc. setW to cover at leaste two full cycles of the longest one 

2 . Use autocorrelation to find the longest cycle : 
plot the autocorrelation function (ACF) of your time series . The first peak above the significance line is the longest cycle . 
what is autocorrelation ? it is a measure of how similar a time series is to itself at different lags . 

3. tune on the validation set : use the validation set to tune the window size . 

4. Beaware of the curse of the dimensionality : if you have a lot of features , you need a lot of data to train a model . a very large W makes each training very high dimensional , so you need a lot of data to train a model . 