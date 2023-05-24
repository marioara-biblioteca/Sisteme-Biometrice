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
    plt.title(wav_file)
    plt.plot(time,data/np.abs(np.max(data)))
    plt.plot(time_index/fs,energy_index/np.max(energy_index))
    plt.xlabel('Timp [s]')
    plt.ylabel('Amplitudine')
    plt.legend(['Energia','Semnal'])
    plt.grid()
    plt.show()
    #din plot alegem pragul
    #ca sa eliminam cadrele care nu contin semnal vocal

def cerinta_2(threshold):
    time_index=[]
    energy_index=[]
    filtered_data=[]
    for i in range(0,L,L_win-L_overlap):
        if i+L_win<L:
            frame=data[i:i+L_win]
            frame_energy=np.matmul(frame,frame.T)
            
            time_index.append(i)
            energy_index.append(frame_energy)
            filtered_data.append(frame)

    energy_index=np.array(energy_index)
    good_index=np.where(energy_index>threshold)[0]
    
    time_index=np.array(time_index)[good_index]
    energy_index=np.array(energy_index)[good_index]
    
    fig=plt.figure()
    #acum ar trebui sa plotam doar partea din audio cu prop specificata
    plt.title(wav_file)
    plt.plot(time,data)
    plt.plot(time_index/fs,energy_index/np.max(energy_index))
    plt.xlabel('Timp [s]')
    plt.ylabel('Amplitudine')
    plt.legend(['Energia','Semnal'])
    plt.grid()
    plt.show()


def cerintele_3_4_5():
    win=np.hamming(L_win)
    n_ceps=128
    filter_bank=librosa.filters.mel(sr=fs,n_fft= nfft, n_mels=128, fmin = 0.0, fmax=fs/2)
    mel_filters=np.concatenate((filter_bank[:,:-1],np.flip(filter_bank[:,:-1])),axis=1)
    filter_bank=filter_bank[:,:-1]
    x=[]
    frame_0=None
    for i in range(0,L,L_win-L_overlap):
        if i+L_win<L:
            frame=data[i:i+L_win]
            if i==0:
                frame_0=frame
            S=np.fft.fft(frame*win,nfft)
            S=S.real*S.real+S.imag*S.imag
            S_mel=np.matmul(S[:int(nfft/2)],filter_bank.T)
            S_log=np.log10(S_mel+0.00001)
            S_ifft=scipy.fftpack.idct(S_log,type=2)
            x.append(np.real(S_ifft))
    mfcc=np.array(x)
    mfcc_lr=librosa.feature.mfcc(y=data, sr=fs,S=None, n_mfcc=n_ceps,dct_type=2)   
    f=np.linspace(0,4000,nfft//2)
    tMFCC=np.linspace(0,127/fs,128)

    # from python_speech_features import mfcc
    n_ceps=12
    frame = frame_0 * np.hamming(len(frame))
    # Aplicăm STFT
    spec = librosa.stft(frame, n_fft=512, hop_length=int(0.010 * fs), win_length=int(0.004 * fs))
    # Aplicăm filtrul băndei Mel
    mel_spec = librosa.feature.melspectrogram(S=np.abs(spec)**2, sr=fs, n_mels=40)
    # Aplicăm logaritmul
    log_mel_spec = librosa.power_to_db(mel_spec)
    # Aplicăm DCT și obținem coeficienții mel-cepstrali
    ceps = librosa.feature.mfcc(S=log_mel_spec, n_mfcc=n_ceps)
    
    # Afisam graficul coeficientilor mel-cepstrali pentru fiecare cadru
    # plt.figure()
    # plt.plot(ceps)
    # plt.title(f"Coeficienti Mel-Cepstrum pentru frame-ul 0 si fisierul {wav_file}")
    # plt.xlabel("Coefficient Index")
    # plt.ylabel("Coefficient Value")
    # plt.show()

    fig=plt.figure()
    plt.title(wav_file)
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
    plt.title(wav_file)
    ax=fig.add_subplot(1,1,1)

    plt.grid()
    plt.xlabel('Timp [s]')
    plt.plot(tMFCC,mfcc_lr[:,15])
    #cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    plt.show()

# cerinta_1()
# cerinta_2(0.0001)
# cerintele_3_4_5()
#obs chiar se vede un impuls

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
wav_file='prop.wav'
data,fs = librosa.load(wav_file)
filter_bank=librosa.filters.mel(sr=fs,n_fft= nfft, n_mels=256  , fmin = 0.0, fmax=fs/2)
# filter_bank=filter_bank[:,:-1]

x=filter_bank[110]
frame_length = int(fs * 0.03)
x = x[:frame_length]

ps = [2, 4, 6, 8]
def exercitiul2(x,ps):
    for p in ps:
        A,E=levinson(x,p)
        w, H = scipy.signal.freqz(1,A,128) 
        H = np.abs(H)
        H = H / np.max(H)
        X=np.abs(np.fft.fft(x,256)) #12 este numarul cadrului (alegeti o valoarea corespunzatoare fisierului pe care-l analizati)
        X=np.abs(X[:128])
        X=X/np.max(X)
        plt.plot(X, H, label=f"p = {p}")

    plt.xlabel("Frecventa normalizata")
    plt.ylabel("Amplitudine normalizata")
    plt.title(f"Caracteristicile de amplitudine ale filtrelor estimate {wav_file}")
    plt.legend()
    plt.show()


exercitiul2(x,ps)