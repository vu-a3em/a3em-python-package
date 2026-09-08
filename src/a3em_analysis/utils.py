import librosa
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt


def preprocess(audio: np.ndarray, sample_rate: int, normalization: float = 0.7):
    # parameter validation
    if normalization < 0.0 or normalization > 1.0:
        raise ValueError('the normalization factor must be between 0.0 and 1.0')

    # low pass filter
    low_pass = butter(5, 800, btype='lowpass', fs=sample_rate, output='sos')
    audio = sosfilt(low_pass, audio)

    # resample to 2000
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=2000)

    # high pass filter
    high_pass = butter(2, 4, btype='highpass', fs=sample_rate, output='sos')
    audio = sosfilt(high_pass, audio)

    # normalize 70%
    audio = audio / np.max(np.abs(audio)) * normalization

    return audio, 2000


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

    # Harmonic-to-noise ratio (HPSS approximation)
    harmonic, percussive = librosa.effects.hpss(audio)

    harmonic_energy = np.sum(harmonic ** 2)
    noise_energy = np.sum(percussive ** 2)

    hnr = 10 * np.log10(
        (harmonic_energy + 1e-10) /
        (noise_energy + 1e-10)
    )

    # Low-frequency harmonic-to-noise ratio
    low_pass = butter(2, 60, btype='lowpass', fs=sample_rate, output='sos')
    low_audio = sosfilt(low_pass, audio)

    harmonic, percussive = librosa.effects.hpss(low_audio)

    harmonic_energy = np.sum(harmonic ** 2)
    noise_energy = np.sum(percussive ** 2)

    hnr_low = 10 * np.log10(
        (harmonic_energy + 1e-10) /
        (noise_energy + 1e-10)
    )

    return {
        'peak_freq': peak_freq,
        'centroid': mean_centroid,
        'bandwidth': mean_bandwidth,
        'freq_5': freq_5,
        'freq_95': freq_95,
        'hnr': hnr,
        'hnr_low': hnr_low,
        **mfcc_dict,
        **mel_dict
    }


def clean_directory(path: Path, base_path: bool = True):
    contents = list(path.iterdir())
    while len(contents) != 0:
        if contents[0].is_dir():
            clean_directory(contents[0], base_path=False)
        else:
            contents[0].unlink()
        contents.pop(0)
    if not base_path:
        path.rmdir()
