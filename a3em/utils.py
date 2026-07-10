import librosa
import numpy as np
import os
from scipy.signal import butter, sosfilt


def preprocess(audio: np.ndarray, sample_rate: int, normalization: float = 0.7):
    # parameter validation
    if normalization < 0.0 or normalization > 1.0:
        raise ValueError('the normalization factor must be between 0.0 and 1.0')

    # high pass filter
    high_pass = butter(2, 15, btype='highpass', fs=sample_rate, output='sos')
    audio = sosfilt(high_pass, audio)

    # low pass filter
    low_pass = butter(2, 200, btype='lowpass', fs=sample_rate, output='sos')
    audio = sosfilt(low_pass, audio)

    # normalize 70%
    audio = audio / np.max(np.abs(audio)) * normalization

    return audio


def extract_features(audio: np.ndarray, sample_rate: int) -> float:
    # spectrograms
    S = np.abs(librosa.stft(audio))**2
    power_per_freq = S.mean(axis=1)
    S_mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=26)

    #peak freq
    freqs  = librosa.fft_frequencies(sr=sample_rate)
    freq_mask = (freqs > 15) & (freqs < 60)
    peak_freq = freqs[freq_mask][np.argmax(power_per_freq[freq_mask])]

    #centroid
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    mean_centroid = centroid.mean()

    # bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    mean_bandwidth = bandwidth.mean()

    # 5% and 95% freqs
    cumulative = np.cumsum(power_per_freq)
    cumulative = cumulative /cumulative[-1]
    freq_5 = freqs[np.searchsorted(cumulative, 0.05)]
    freq_95 = freqs[np.searchsorted(cumulative, 0.95)]

    # MFCCS
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_dict = {f'mfcc_{i+1}': mfccs[i].mean() for i in range(13)}

    mel_means = S_mel.mean(axis=1)
    mel_dict = {f'mel_mean_{i+1}': v for i, v in enumerate(mel_means)}

    # TODO - spectral flatness

    return {
        'peak_freq': peak_freq,
        'centroid': mean_centroid,
        'bandwidth': mean_bandwidth,
        'freq_5': freq_5,
        'freq_95': freq_95,
        **mfcc_dict,
        **mel_dict
    }


def __get_path_stem(path: str) -> str:
    split_char = '/' if os.name == 'posix' else '\\'
    directories = path.split(split_char)
    file_name = directories[-1]
    parts = file_name.split('.')
    return parts[0]