# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:55:20 2020

@author: Stefan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:57:30 2020

@author: Stefan
"""

from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import librosa# im1port sounddevice as sd
import scipy
from matplotlib.widgets import Cursor

soundFolder="D:/Teaching/Biometrice/Cursuri/Licenta_OK/Online/Laboratoare/sunete/"
file_name=soundFolder+"o_e.wav"


sr,s = read(file_name)


# sd.play(s, sr)
# status = sd.wait()  # Wait until file is done playing
s=s.astype('float')
b=[1 -0.95]
a=1
s=scipy.signal.lfilter(b,a,s)

# sr=float(sr)
L=len(s) #lungimea semnalului (nr de esantioane)
L2=int(L/2) #jumatate din lungimea semnalului
t=np.linspace(0,(L-1)/sr,L)
Nfft=2048;
Nfft2=int(Nfft/2)
L_win=int(0.040*sr); #lungimea ferestrei 40ms
L_overlap=int(0.015*sr); #lungimea suprapunere 15ms
    
win=np.hamming(L_win)
i=0
N=128 #numarul de filtre pe scara Mel
filter_bank=librosa.filters.mel(sr, Nfft, n_mels=N, fmin = 0.0, fmax=sr/2)
# mel_filters=np.concatenate((filter_bank[:,0:-1:1],filter_bank[:,-1:0:-1]),axis=1)
mel_filters=np.concatenate((filter_bank[:,:-1],np.flip(filter_bank[:,:-1])),axis=1)
filter_bank=filter_bank[:,:-1]
fig=plt.figure()
plt.plot(mel_filters.transpose())
plt.grid()  
plt.show()

# fig=plt.figure()
# plt.plot(filter_bank.transpose())
# plt.grid()  
# plt.show()

x=[]
timeTmp=[]
tmpE=[]
s=s/np.max(np.abs(s))
while i<L:
    if (i+L_win<L):
        s_win=s[i:i+L_win]
        frameE=np.matmul(s_win,s_win.T)
        S=np.fft.fft(s_win*win,Nfft)
        S=S.real*S.real+S.imag*S.imag
        S_mel=np.matmul(S[:int(Nfft/2)],filter_bank.T)
        S_log=np.log10(S_mel+0.00001)
        # S_log=np.concatenate((S_log[:],np.flip(S_log)))
        # S_ifft=np.fft.ifft(S_log)
        S_ifft=scipy.fftpack.idct(S_log,type=2)
        x.append(np.real(S_ifft))
        timeTmp.append(i)
        tmpE.append(frameE)
    i=i+L_win-L_overlap
    
# fig=plt.figure()
# plt.plot(S_ifft*128)
# plt.grid()  
# plt.show()

mfcc=np.array(x)
timeStamp=np.array(timeTmp)
E=np.array(tmpE)

mfcc_lr=librosa.feature.mfcc(s, sr=sr,S=None, n_mfcc=128,dct_type=2)
# (y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0, **kwargs
f=np.linspace(0,4000,Nfft2)
tMFCC=np.linspace(0,127/sr,128)
    
fig=plt.figure()
fig.add_subplot(2,1,1)
plt.plot(tMFCC,mfcc[50,:128]),
plt.grid()
plt.xlabel('Timp [s]')
fig.add_subplot(2,1,2)
plt.plot(tMFCC,mfcc_lr[:,15])
plt.xlabel('Timp [s]')
plt.title('calculat cu librosa')
plt.grid()

fig=plt.figure()
ax=fig.add_subplot(1,1,1)

plt.grid()
plt.xlabel('Timp [s]')
plt.plot(tMFCC,mfcc_lr[:,15])
cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
plt.show()
    
fig=plt.figure()
plt.plot(t,s/np.abs(np.max(s)))
plt.plot(timeStamp/sr,E/np.max(E))
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')
plt.legend(['Energia','Semnal'])
plt.grid()
plt.show()

