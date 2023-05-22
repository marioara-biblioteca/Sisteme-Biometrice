import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display

def accentuare(s):
    b=[1 -0.99] 
    a=[1] 
    s=scipy.signal.lfilter(b,a,s) 
    s=s/np.max(np.abs(s)) #normat
    return s

wav_file='prop.wav'

fs, data = wavfile.read(wav_file)
data=data.astype('float')
data=accentuare(data)#il si normalizeaza

L=len(data)
time = np.linspace(0,(L-1)/fs,L)

nfft=2048
L_win=int(0.04*fs) #lungimea ferestrei 40ms
L_overlap=int(0.015*fs) # lungimea suprapunere 15ms


def cerinta_1():
    time_index=[]
    energy_index=[]
    for i in range(0,L,L_win-L_overlap):
        if i+L_win<L:
            frame=data[i:i+L_win]
            frame_energy=np.matmul(frame,frame.T)
            
            time_index.append(i)
            energy_index.append(frame_energy)

    time_index=np.array(time_index)
    energy_index=np.array(energy_index)

    fig=plt.figure()
    plt.plot(time,data/np.abs(np.max(data)))
    plt.plot(time_index/fs,energy_index/np.max(energy_index))
    plt.xlabel('Timp [s]')
    plt.ylabel('Amplitudine')
    plt.legend(['Energia','Semnal'])
    plt.grid()
    plt.show()
    #din plot alegem pragul
    #ca sa eliminam cadrele care nu contin semnal vocal

def cerinta_2():
    time_index=[]
    energy_index=[]
    filtered_data=[]
    for i in range(0,L,L_win-L_overlap):
        if i+L_win<L:
            frame=data[i:i+L_win]
            frame_energy=np.matmul(frame,frame.T)
            if frame_energy>10.0:
                time_index.append(i)
                energy_index.append(frame_energy)
                filtered_data.append(frame)

    time_index=np.array(time_index)
    energy_index=np.array(energy_index)
    
    fig=plt.figure()
    #acum ar trebui sa plotam doar partea din audio cu prop specificata
    plt.plot(time,data)
    plt.plot(time_index/fs,energy_index/np.max(energy_index))
    plt.xlabel('Timp [s]')
    plt.ylabel('Amplitudine')
    plt.legend(['Energia','Semnal'])
    plt.grid()
    plt.show()

# cerinta_1()
# cerinta_2()

def cerintele_3_4_5():
    win=np.hamming(L_win)
    
    filter_bank=librosa.filters.mel(sr=fs,n_fft= nfft, n_mels=128, fmin = 0.0, fmax=fs/2)
    mel_filters=np.concatenate((filter_bank[:,:-1],np.flip(filter_bank[:,:-1])),axis=1)
    filter_bank=filter_bank[:,:-1]
    x=[]
    for i in range(0,L,L_win-L_overlap):
        if i+L_win<L:
            frame=data[i:i+L_win]
            S=np.fft.fft(frame*win,nfft)
            S=S.real*S.real+S.imag*S.imag
            S_mel=np.matmul(S[:int(nfft/2)],filter_bank.T)
            S_log=np.log10(S_mel+0.00001)
            S_ifft=scipy.fftpack.idct(S_log,type=2)
            x.append(np.real(S_ifft))
    mfcc=np.array(x)
    mfcc_lr=librosa.feature.mfcc(y=data, sr=fs,S=None, n_mfcc=128,dct_type=2)   
    f=np.linspace(0,4000,nfft//2)
    tMFCC=np.linspace(0,127/fs,128)
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
    #cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    plt.show()
cerintele_3_4_5()
#obs chiar se vede un impuls