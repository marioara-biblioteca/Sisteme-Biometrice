# import matplotlib.pyplot as plt
# import numpy as np
# import wave
# import sys

# def for_file(file,ax):
#     spf = wave.open(file, "r")

#     # Extract Raw Audio from Wav File
#     signal = spf.readframes(-1)
#     signal = np.fromstring(signal, dtype=np.int16)

#     num_frames=spf.getnframes()
#     sample_rate = spf.getframerate()

#     duration = num_frames/ float(sample_rate)
#     # If Stereo
#     if spf.getnchannels() == 2:
#         print("Just mono files")
#         sys.exit(0)


#     Time = np.linspace(0, len(signal) / sample_rate, num=len(signal))

    
#     ax.plot(Time, signal)
    
# fig, (ax1, ax2,ax3) = plt.subplots(3)
# ax1.set_title('Aaa.wav')
# ax2.set_title('sare.wav')
# ax3.set_title('prop.wav')
# for_file('aaa.wav',ax1)
# for_file('sare.wav',ax2)
# for_file('prop.wav',ax3)
# plt.show()

from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt
def for_one_file(filename,ax1,ax2,ax3):
    fs_rate, signal = wavfile.read(filename)
    print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    print ("Channels", l_audio)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print ("secs", secs)
    Ts = 1.0/fs_rate # sampling interval in time
    print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    FFT = abs(scipy.fftpack.fft(signal))
    FFT_side = FFT[range(N//2)] # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(N//2)] # one side frequency range
    fft_freqs_side = np.array(freqs_side)

    ax1.plot(t, signal, "g") # plotting the signal
    ax1.set_title('signal')
    ax2.plot(freqs, FFT, "r") # plotting the complete fft spectrum
    ax2.set_title('complete fft spectrum')
    
    ax3.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
    ax3.set_title('positive fft spectrum')


fig, (ax1, ax2,ax3) = plt.subplots(3)
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Count dbl-sided')
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Count sngle-sided')
# for_one_file('aaa.wav',ax1,ax2,ax3)
# for_one_file('sare.wav',ax1,ax2,ax3)
for_one_file('prop.wav',ax1,ax2,ax3)
plt.show()