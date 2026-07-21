import a3em.utils
import librosa
import os
import random
import requests
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

BUFFER = 0.2
DEFAULT_PATH='./data/'

def load_data(path: Path = DEFAULT_PATH, sample_rate: int = 2000, test_split: float = 0.25,random_state: int = 123,shuffle: bool = True) -> tuple:
    _, df = load_clips(path, sample_rate=sample_rate)
    labels = df['quality'].replace({ 0: 0, 2: 1, 3: 1, 4: 1 })
    labels.name = 'label'
    drop = ['call_type', 'quality', 'earflap', 'source', 'sample_count', 'sample_rate', 'start_sample', 'end_sample', 'overlap']
    data = df.drop(columns=drop)
    return data, labels if test_split == 0.0 else train_test_split(data, labels, test_size=test_split, random_state=random_state, shuffle=shuffle)
    

# TODO - add progress bar
def load_clips(path: Path = DEFAULT_PATH, rumble_only: bool = False, noise_seed: int = None, sample_rate: int = 2000) -> tuple:
    clips, rows = [], []
    for clip, row in iterclip(path, rumble_only, noise_seed, sample_rate):
        clips.append(clip)
        rows.append(row)
    return clips, pd.DataFrame(rows)


def iterclip(path: Path = DEFAULT_PATH, rumble_only: bool = False, noise_seed: int = None, sample_rate: int = 2000):
    print('loading audio data')
    audio_files, annotation_files = __prefetch(path)
    audio_dict = {}
    for file in audio_files: 
        audio_dict[file.stem] = file  
   
    print('extracting clip')
    clips, all_rows = [], []
    for file in annotation_files:
        audio, _ = librosa.load(audio_dict[file.stem], sr=sample_rate)
        annotations = pd.read_csv(file, sep='\t')
        metadata = __extract_rumbles(annotations, sample_rate, file.stem, len(audio))
        if len(metadata) == 0:
            continue

        if not rumble_only:
            background_noise_metadata = __extract_background_noise(metadata, sample_rate, file.stem, len(audio))
            metadata = pd.concat([metadata, background_noise_metadata], ignore_index=True)
        metadata = __filter_rumbles(metadata)
        metadata = metadata.sample(frac=1, random_state=noise_seed).reset_index(drop=True)

        for i in range(len(metadata)):
            row = metadata.iloc[i]
            clip = audio[row['start_sample']: row['end_sample']]
            clips.append(clip)
            all_rows.append(__extract_clip_features(clip, sample_rate, row.to_dict()))

    print('extraction complete')
    for clip, row in list(zip(clips, all_rows)):
        yield clip, row


def __prefetch(path: Path) -> tuple:
    annotations_path, audio_path = path.joinpath('manualAnnotations'), path.joinpath('audiomoth')
    if not __validate_data_path(path): 
        a3em.utils.clean_directory(path)
        __download_data(path)
    print('prefetch complete')
    return sorted(audio_path.glob('*.WAV')), sorted(annotations_path.glob('*.txt'))


def __validate_data_path(path: Path) -> bool:
    print('checking for existing files')
    annotations_path, audio_path = path.joinpath('manualAnnotations'), path.joinpath('audiomoth')
    if not (annotations_path.is_dir() and audio_path.is_dir()):
        print('local data missing or incomplete')
        return False
    
    # TODO - add more validation

    return True


def __extract_clip_features(clip: np.ndarray, sample_rate: int, row: dict) -> dict:
    clip_processed = a3em.utils.preprocess(clip, sample_rate)
    features = a3em.utils.extract_features(clip_processed, sample_rate)
    return { **row, ** features }


# TODO - modify to use API with user provided API key
def __download_data(path):
    print('fetching data')
    annotation_request = requests.get(os.getenv('LABELS_DOWNLOAD'))
    annotation_zip = path.joinpath('annotations.zip')
    with open(annotation_zip, 'wb') as fd:
        for chunk in annotation_request.iter_content(chunk_size=128):
            fd.write(chunk)
    audio_request = requests.get(os.getenv('AUDIO_DOWNLOAD'))
    audio_zip = path.joinpath('audio.zip')
    with open(audio_zip, 'wb') as fd:
        for chunk in audio_request.iter_content(chunk_size=128):
            fd.write(chunk)

    print('unpacking')
    with zipfile.ZipFile(audio_zip, "r") as zip_ref:
        zip_ref.extractall(path)
    with zipfile.ZipFile(annotation_zip, "r") as zip_ref:
        zip_ref.extractall(path)
    audio_zip.unlink()
    annotation_zip.unlink()


def __extract_rumbles(annotations: pd.DataFrame, sample_rate: int, audio_source: str,audio_length: int) -> pd.DataFrame:
    rows, drop = [], ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)']
    for i in range(len(annotations)):
        row = annotations.iloc[i].to_dict()
        row['source'] = audio_source
        row['sample_count'] = audio_length
        row['sample_rate'] = sample_rate
        row['start_sample'] = max(0, int((row['Begin Time (s)'] - BUFFER) * sample_rate))
        row['end_sample'] = min(int((row['End Time (s)'] + BUFFER) * sample_rate), audio_length)
        for key in drop:
            del row[key]
        rows.append(row)
    return pd.DataFrame(rows)


def __extract_background_noise(rumble_annotations: pd.DataFrame, sample_rate: int, audio_source: str, audio_length: int) -> pd.DataFrame:
    rows = []
    rumble_event_ranges = list(zip(rumble_annotations['start_sample'], rumble_annotations['end_sample']))

    durations = list(map(lambda time: time[1] - time[0], rumble_event_ranges))
    duration_mean = np.mean(durations)
    duration_standard_deviation = np.std(durations)

    clip_count = len(rumble_annotations)
    clip_ranges = []
    while len(clip_ranges) < clip_count:
        center = int(audio_length * random.random())
        clip_length = int(duration_mean + (duration_standard_deviation * random.random() * random.randrange(-1, 2, 2)))
        clip_range = max(0, int(center - clip_length * 0.5)), min(int(center + clip_length * 0.5), audio_length)
        if __range_is_background_noise(clip_range, rumble_event_ranges):
            clip_ranges.append(clip_range)

    for clip_start, clip_end in clip_ranges:
        rows.append({
            'call_type': 'BKG',
            'quality': 0,
            'overlap': 'N',
            'earflap': 0,
            'source': audio_source,
            'sample_count': clip_end - clip_start,
            'sample_rate': sample_rate,
            'start_sample': clip_start,
            'end_sample': clip_end
        })

    return pd.DataFrame(rows)


    
def __range_is_background_noise(clip_range: tuple, rumble_ranges: list) -> bool:
    (clip_start, clip_end) = clip_range
    for rumble_start, rumble_end in rumble_ranges:
        if (clip_start > rumble_start and clip_start < rumble_end) or (clip_end > rumble_start and clip_end < rumble_end):
            return False
        if clip_start < rumble_end and clip_end > rumble_start:
            return False
    return True


def __filter_rumbles(df: pd.DataFrame) -> pd.DataFrame:
    df['earflap'] = pd.to_numeric(df['earflap'], errors='coerce')
    return df[
        (df['call_type'].isin(['RUM', 'BKG'])) &
        (df['earflap'].isin([0, 1])) &
        (df['overlap']  == 'N') & # TODO - change from 'N' to 0
        (df['quality'].isin([0, 2, 3, 4]))
    ]
