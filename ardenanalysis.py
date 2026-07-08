from datetime import datetime, timedelta
import librosa
import matplotlib as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfilt

BUFFER = 0.2

class AudioAnalysis:
    @staticmethod
    def preprocess(clip: np.ndarray, sr: int) -> np.ndarray:
        # high pass filter
        high_pass = butter(2, 15, btype='highpass', fs=sr, output='sos')
        clip = sosfilt(high_pass, clip)

        # low pass filter
        low_pass = butter(2, 200, btype='lowpass', fs=sr, output='sos')
        clip = sosfilt(low_pass, clip)

        # normalize 70%
        clip = clip / np.max(np.abs(clip)) * 0.7

        return clip
    
    
    @staticmethod
    def feature_extract(clip: np.ndarray, sr: int) -> dict:
        # spectrograms
        S = np.abs(librosa.stft(clip))**2
        power_per_freq = S.mean(axis=1)
        S_mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=26)

        #peak freq
        freqs  = librosa.fft_frequencies(sr=sr)
        freq_mask = (freqs > 15) & (freqs < 60)
        peak_freq = freqs[freq_mask][np.argmax(power_per_freq[freq_mask])]

        #centroid
        centroid = librosa.feature.spectral_centroid(y=clip, sr=sr)
        mean_centroid = centroid.mean()

        # bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=clip, sr=sr)
        mean_bandwidth = bandwidth.mean()

        # 5% and 95% freqs
        cumulative = np.cumsum(power_per_freq)
        cumulative = cumulative /cumulative[-1]
        freq_5 = freqs[np.searchsorted(cumulative, 0.05)]
        freq_95 = freqs[np.searchsorted(cumulative, 0.95)]

        # MFCCS
        mfccs = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13)
        mfcc_dict = {f'mfcc_{i+1}': mfccs[i].mean() for i in range(13)}

        mel_means = S_mel.mean(axis=1)
        mel_dict = {f'mel_mean_{i+1}': v for i, v in enumerate(mel_means)}

        # ADD
        # spectral flatness

        return {
            'peak_freq': peak_freq,
            'centroid': mean_centroid,
            'bandwidth': mean_bandwidth,
            #'duration': duration,
            'freq_5': freq_5,
            'freq_95': freq_95,
            **mfcc_dict,
            **mel_dict
        }
    

class DataDriver:
    def __init__(self, annotation_path: Path, audio_path: Path, fig_path: Path):
        self._annotation_files = sorted(annotation_path.glob("*.txt"))
        self._audio_path = audio_path
        self._fig_path = fig_path


    def process(self) -> pd.DataFrame:
        all_rows = []
        for ann_path in self._annotation_files:
            combined = self._process_annotation(ann_path)
            all_rows += combined
        return pd.DataFrame(all_rows)
    

    def _process_annotation(self, ann_path: Path) -> list:
        combined = []

        # only include high quality rumbles
        df = self._get_high_quality_rumbles(ann_path)

        # parse recording start time from filename
        parts = ann_path.stem.split('_')
        rec_start = datetime.strptime(parts[1] + parts[2], '%Y%m%d%H%M%S')

        # load matching audio
        wav_path = self._audio_path / (ann_path.stem + '.wav')
        audio, sr = librosa.load(wav_path, sr=2000)

        # loop over each detection in file
        for i in range(len(df)):
            row = df.iloc[i]
            features = self._extract_features_from_clip(row, audio, rec_start, sr, ann_path)
            combined.append(features)
        
        return combined
    

    def _get_high_quality_rumbles(self, ann_path: Path) -> pd.DataFrame:
        df = pd.read_csv(ann_path, sep='\t')
        df['earflap'] = pd.to_numeric(df['earflap'], errors='coerce')
        return df[
            (df['call_type'] == 'RUM') &
            (df['earflap'].isin([0])) &
            (df['quality'].isin([3, 4])) &
            (df['overlap']  == 'N')
        ]
    

    def _extract_features_from_clip(self, clip_info: pd.Series, audio: np.ndarray, rec_start: datetime, sr: int, ann_path: Path) -> dict:
        clip_processed, _ = self._preprocess_clip(clip_info, audio, rec_start, sr)
        features = AudioAnalysis.feature_extract(clip_processed, sr)
        return {
            'filename': ann_path.stem,
            'rec_start': rec_start,
            'abs_begin': rec_start + timedelta(seconds=clip_info['Begin Time (s)']),
            'abs_end': rec_start + timedelta(seconds=clip_info['End Time (s)']),
            'duration': clip_info['End Time (s)'] - clip_info['Begin Time (s)'],
            **clip_info.to_dict(),
            **features,
        }


    def _preprocess_clip(self, clip_info: pd.Series, audio: np.ndarray, rec_start: datetime, sr: int) -> tuple:
        start_sample = int((clip_info['Begin Time (s)'] - BUFFER) * sr)
        end_sample = int((clip_info['End Time (s)'] + BUFFER) * sr)
        start_time = rec_start + timedelta(seconds = clip_info['Begin Time (s)'])
        start_time.strftime('%Y%m%d_%H%M%S')
        clip = audio[start_sample:end_sample]
        clip_processed = AudioAnalysis.preprocess(clip, sr)
        return (clip_processed, start_time)


    def save_images(self):
        for ann_path in self._annotation_files:
            # only include high quality rumbles
            df = self._get_high_quality_rumbles(ann_path)

            # parse recording start time from filename
            parts = ann_path.stem.split('_')
            rec_start = datetime.strptime(parts[1] + parts[2], '%Y%m%d%H%M%S')

            # load matching audio
            wav_path = self._audio_path / (ann_path.stem + '.wav')
            audio, sr = librosa.load(wav_path, sr=2000)

            for i in range(len(df)):
                clip_info = df.iloc[i]
                clip_processed, start_time = self._preprocess_clip(clip_info, audio, rec_start, sr)
                features = AudioAnalysis.feature_extract(clip_processed, sr)
                self._save_image(clip_processed, sr, features, start_time)


    def _save_image(self, clip_processed: np.ndarray, sr: int, features: dict, start_time: datetime):
        spec_proc = librosa.amplitude_to_db(np.abs(librosa.stft(clip_processed, n_fft=1024, hop_length=128)), ref=np.max)
        fig, ax = plt.subplots()
        img_proc = librosa.display.specshow(spec_proc, sr=sr, hop_length=128, x_axis='time', y_axis='hz', ax=ax)
        plt.ylim(0,200)

        peak_freq = features['peak_freq']
        freq_5 = features['freq_5']
        freq_95 = features['freq_95']
        
        ax.axhline(features['peak_freq'],color='white',linestyle='--',label=f'peak_freq {peak_freq}')
        ax.axhline(features['freq_5'],color='blue',linestyle='-.',label=f'freq_5 {freq_5}')
        ax.axhline(features['freq_95'],color='red',linestyle='-',label=f'freq_95 {freq_95}')
        ax.legend(fontsize=8, loc='upper right')

        # make sure the path exists
        fig_path = os.getenv('FIG_PATH')
        fig.savefig(os.path.join(fig_path, f'{start_time}_spectrogram.png'), dpi=150,bbox_inches='tight')

        plt.close(fig)