what Nyquist means when you build models ?
1. Your sampling rate determines what you can learn , 
if you sampel sensor data at 10 hz , you can never detect events that happen faster than 5hz - no matter how powerful you model is 

2. Downsampling destroyes information 
If you aveerage your data into hourly buckets because it is easies you have thrown away all sub hourly patterns permanannetnly make decision consciously 

3. Irregular sampling is dangerous  
The theorem assumes uniform sampling missing values , dropped packets or irregular polling intervals create frequencey ambiguities that can appear as spurious patterns 

4.Upsampling does not add infomration > interporlating your 10 hz data to 100 hz and feeding it a model does not let the model see frequencies between 5hz and 50 hz . You are just creating synthectic data points derived from the iorginal ones . 