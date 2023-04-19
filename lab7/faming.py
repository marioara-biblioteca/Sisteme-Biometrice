import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
import matplotlib.pyplot as plt
import librosa
def framing(wav_file, fs=16000, win_len=0.025, win_hop=0.01):
    fs_rate, signal = wavfile.read(wav_file)
 
    # compute frame length and frame step (convert from seconds to samples)
    frame_length = win_len * fs
    frame_step = win_hop * fs
    signal_length = len(signal)
    frames_overlap = frame_length - frame_step

    # Make sure that we have at least 1 frame+
    num_frames = np.abs(signal_length - frames_overlap) // np.abs(frame_length - frames_overlap)
    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
   
    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    if rest_samples != 0:
        pad_signal_length = int(frame_step - rest_samples)
        z = np.zeros((pad_signal_length))
        pad_signal = np.append(signal, z)
        num_frames += 1
    else:
        pad_signal = signal

    # make sure to use integers as indices
    frame_length = int(frame_length)
    frame_step = int(frame_step)
    num_frames = int(num_frames)

    # compute indices
    idx1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
    idx2 = np.tile(np.arange(0, num_frames * frame_step, frame_step),
                (frame_length, 1)).T
    indices = idx1 + idx2
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

frames = framing('sare.wav', fs=16000, win_len=0.025, win_hop=0.01)
# https://towardsdatascience.com/a-step-by-step-guide-to-speech-recognition-and-audio-signal-processing-in-python-136e37236c24
def fft_with_hamming(wav_file):
    fs_rate, signal = wavfile.read(wav_file)
    M=len(signal)
    hamming=np.hamming(M)
    plt.subplot(3,1,1) 
    plt.plot(hamming,'-k')
    plt.plot(hamming,'o') 
    plt.title('Hamming Window')
    plt.xlabel('Time (samples)') 
    plt.ylabel('Amplitude') 
    transp=np.transpose(signal)
    frames_ham=transp*hamming
    ham_fft=np.fft.fft(frames_ham)
    ham_fft=abs(ham_fft[0:np.ceil((M + 1) / 2.0).astype(np.int)]) / M
    ham_fft**=2
    plt.subplot(3,1,2) 
    plt.title('ABS ^ 2 FFT ')
    plt.xlabel('Frequency') 
    plt.ylabel('Magnitude') 
    plt.plot(ham_fft)
    ######################
    if M%2==0:
        ham_fft[1:len(ham_fft)-1]*=2
    else:
        ham_fft[1:len(ham_fft)]*=2
    exp_ham_fft=10*np.log10(ham_fft)
    plt.subplot(3,1,3) 
    plt.title("Signal's strenth in decibels")
    # x_axis = np.arange(0, np.ceil((M + 1) / 2.0).astype(np.int), 1) * (16000 / M) / 1000.0
    # plt.plot(x_axis,exp_ham_fft)
    plt.plot(exp_ham_fft)
    plt.xlabel('Frequency') 
    plt.ylabel('Magnitude') 

    plt.show()
    
def with_librosa(wav_file):
    fig, ax = plt.subplots()
    y, sr = librosa.load(librosa.ex('trumpet'))
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure()
    librosa.display.specshow(S_db)
    plt.colorbar()
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, ax=ax)
    fig.colorbar(img, ax=ax)
    plt.show()

def mel(wav_file):
    y, sr = librosa.load(librosa.ex('trumpet'))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=16000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()
mel('sare.wav')


