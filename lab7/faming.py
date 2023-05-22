import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import librosa
def framing(wav_file, fs=16000, win_len=0.025, win_hop=0.01):
    fs_rate, signal = wavfile.read(wav_file)
    # plt.plot(signal)
    # plt.ylabel("Amplitude")
    # plt.xlabel("Time")
    # plt.show()
   
    frame_length = win_len * fs
    frame_step = win_hop * fs
    signal_length = len(signal)
    frames_overlap = frame_length - frame_step

    num_frames = np.abs(signal_length - frames_overlap) // np.abs(frame_length - frames_overlap)
    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
   
  
    if rest_samples != 0:
        pad_signal_length = int(frame_step - rest_samples)
        z = np.zeros((pad_signal_length))
        pad_signal = np.append(signal, z)
        num_frames += 1
    else:
        pad_signal = signal

    
    frame_length = int(frame_length)
    frame_step = int(frame_step)
    num_frames = int(num_frames)

    
    idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
    idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step),
                (frame_length, 1)).T
    indices = idx1 + idx2
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames
from scipy import signal

# plt.plot(frames[101,:])
# plt.show()
def fft_with_hamming(wav_file):
    #fs_rate, signal = wavfile.read(wav_file)
    signals=framing(wav_file, fs=16000, win_len=0.025, win_hop=0.01)
    
    M=signals.shape[1]
    hamming=np.hamming(M)
   
    for i in range(190):
        signal_np=signals[i,:]
    
        transp=np.transpose(signal_np)
        frames_ham=transp*hamming
        f,t,Zxx=signal.stft(frames_ham,16000,nperseg=1000)
        # plt.pcolormesh(t, f, np.abs(Zxx)**2, vmin=0, vmax=2 * np.sqrt(2), shading='gouraud')
        # plt.title('STFT Magnitude '+wav_file)
        # plt.ylabel('Frequency [Hz]')
        # plt.xlabel('Time [sec]')
        # plt.show()
        
        
        exp_ham_fft=10*np.log10(np.abs(Zxx)**2)
        plt.title("Signal's strenth in decibels" + wav_file)
        plt.pcolormesh(t, f, exp_ham_fft, vmin=0, vmax=2 * np.sqrt(2), shading='gouraud')
        plt.xlabel('Frequency') 
        plt.ylabel('Magnitude') 
        plt.show()

        
    
    
def with_librosa(wav_file):
    fig, ax = plt.subplots()
    y, sr = librosa.load(wav_file)
    D = librosa.stft(y,n_fft=2048,hop_length=512)  
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure()
    librosa.display.specshow(S_db)
    plt.colorbar()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, ax=ax)
    fig.colorbar(img, ax=ax)
    plt.title('Librosa '+wav_file)
    plt.show()
#separa audioul
def filter_bank_wav(wav_file):
    y, sr = librosa.load(wav_file)
    filter_bank = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=128, fmin=0, fmax=sr / 2)
    return filter_bank

def mel(wav_file):
    y, sr = librosa.load(wav_file)
    samples=filter_bank_wav(wav_file)
    for sample in samples:
        S = librosa.feature.melspectrogram(y=sample, sr=sr, n_mels=128,
                                        fmax=16000)
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr,
                            fmax=16000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram '+wav_file)
        plt.show()






# mel('aaa.wav')
# mel('sare.wav')
mel('whistle.wav')
# with_librosa('sare.wav')
# fft_with_hamming('sare.wav')
# fft_with_hamming('sare.wav')
# fft_with_hamming('aaa.wav')