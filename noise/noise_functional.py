

import soundfile as sf
import librosa
import numpy as np
from scipy import signal
import IPython

def generate_noise():
    """
    Grenerates the noise.
    The window size, steps were choosen empirically.
    To add the randomnes on the noise enough to vary the param.
    of the distributions.
    return: noise freq. vec
    
    """
    
    fs = 1024
    N = 10*fs
    
    n_1 = np.random.normal(0,0.1,N)
    n_2 = np.random.uniform(0,0.1,N)
    n_3 = np.random.laplace(0,0.05,N)
    y_1,y_2,y_3 = n_1,n_2,n_3
    
    _, _, Z_1 = signal.stft(y_1, fs, nperseg=1024)
    _, _, Z_2 = signal.stft(y_2, fs, nperseg=1024)
    _, _, Z_3 = signal.stft(y_3, fs, nperseg=1024)
    
    noise_1 = np.mean(Z_1,axis = 1)
    noise_2 = np.mean(Z_2,axis = 1)
    noise_3 = np.mean(Z_3,axis = 1)
    
    step = 170
    noise_vec = np.concatenate((noise_1[:step],
                                noise_2[step:2*step],
                                noise_3[2*step:3*step+3]))
    return noise_vec



def add_noise(path):
    """
    Add noise to the audio file
    input: path to the file
    return: waveform of the audio and samp.rate
    
    """
    
    data, samplerate = librosa.load(path)
    
    window = 1024
    _, _, Z = signal.stft(data, samplerate, nperseg=window)
    
    noise_vec = generate_noise()
    Z = Z + np.array([noise_vec for i in range(Z.shape[1])]).T
    
    _, data_noise = signal.istft(Z, samplerate, nperseg=window)
    return data_noise,samplerate






