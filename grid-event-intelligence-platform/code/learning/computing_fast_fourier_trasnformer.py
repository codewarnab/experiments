import numpy as np
import matplotlib.pyplot as plt

#simulate a simple signal : 50 hz a fault at 250 Hz 
fs = 1000 # sampling frequency  1000 samples per second 
N=  1024 # number of samples 
t = np.arange(N) / fs  # TIME axis  in seconds 

#clean 50 hz power + 250 hz fault compoenent (samll amplitude)
signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 250 * t)

#compute fft 
fft = np.fft.rfft(signal) # rfft for real values signals 
freq = np.fft.rfftfreq(N, 1/fs) # frequency axis 


magnitude = np.abs(fft) # we care about the magnitude of the signal 

plt.figure(figsize=(10, 5))
plt.plot(freq, magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT of Signal')
plt.show()

# what practtionaers actually look at 
# In a power system you rarely look at the raw fft magnitude Instead you look for 
#Harmonic distortions : energy at multiples of 50 / 60 hz Transformer saturating or non linear load casuse this 
#Inter harmonic energy at non integer multipels caused by variable speed drives 
#sub harmonics : energy below 50 hz can indicate ferroresonce - a dangerous condition 
#characterics faul frequeuncies : 60 hz , 120 hz , 180 hz , etc. 