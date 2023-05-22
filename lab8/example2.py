# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 14:57:25 2020

@author: Stefan
"""

from scipy.io.wavfile import read
from scipy.signal import freqz
import scipy
import matplotlib.pyplot as plt
import numpy as np
#import librosa

def main():
    file_name="F:\Teaching\Biometrice\Cursuri\Licenta_OK\Online\Laboratoare\sunete\o_e.wav"

    sr,s=read(file_name)
    s=np.asarray(s,'float')
    s=s/np.max(np.abs(s))
    
    b=[1 -0.99]
    a=[1]
    s=scipy.signal.lfilter(b,a,s)
    s=s/np.max(np.abs(s))

    L=len(s)
    Nfft=2048
    Nfft2=int(Nfft/2)
    win=int(0.030*sr)
    over=int(0.015*sr)
    
    #ordinul de predictie
    p=50
    i=0;
    x=[]
    while(i<L):
        if (i+win<L):
            s_win=s[i:i+win]
            x.append(s_win)
        i=i+win-over
    
    A,E=levinson(x[120],p)


    w, H = freqz(1,A,120)
    H=np.abs(H)
    H=H/np.max(H)
    X=np.abs(np.fft.fft(x[120]))
    X=np.abs(X)
    X=X/np.max(X)
    

    fig=plt.figure()
    plt.plot(w,X[:int(len(X)/2)])
    plt.plot(w,H)
    plt.xlabel('Frecventa [rad]')
    plt.ylabel('Amplitudine')
    # plt.title('Filtrul de predictie liniara, ordin de predictie p=12')
    plt.legend(['Spectrul semnalului vocal','Filtrul de predictie liniara'])
    plt.grid()
    plt.show()
    ###PLP 
    
def levinson(s,p):
    r=np.correlate(s,s,"full")
    L=len(r)
    mid=int((L-1)/2)
    Ak=np.zeros((2,))
    aux=np.zeros((1,))
    Ek=np.zeros((1,))
    Ak[0]=1
    R=r[mid:]
    Ak[1]=-R[1]/R[0]
    Ek=R[0]+R[1]*Ak[1]
    for i in range(2,p+1):
        var_lambda=0
        temp=0
        for j in range(0,i):
            temp=temp-Ak[j]*R[i-1+1-j]
        var_lambda=temp/Ek
        U=np.concatenate((Ak,aux))
        V=np.flip(U)
        Ak=U+var_lambda*V
        Ek=(1-var_lambda*var_lambda)*Ek
    return Ak,Ek
    
    
if __name__=="__main__":
    main()