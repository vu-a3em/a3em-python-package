import a3em.utils
import librosa
import os
import random
import requests
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BUFFER = 0.2
DEFAULT_PATH='./data/'

def load_data(path: Path = DEFAULT_PATH, sample_rate: int = 2000, test_split: float = 0.25, random_state: int = 123, shuffle: bool = True) -> tuple:
    _, df = load_clips(path, sample_rate=sample_rate, random_state=random_state)
    labels = df['quality'].replace({ 0: 0, 2: 1, 3: 1, 4: 1 })
    labels.name = 'label'
    drop = ['call_type', 'quality', 'earflap', 'source', 'sample_rate', 'Begin Time (s)', 'End Time(s)', 'overlap']
    data = df.drop(columns=drop)
    return data, labels if test_split == 0.0 else train_test_split(data, labels, test_size=test_split, random_state=random_state, shuffle=shuffle)
    

# TODO - add progress bar
def load_clips(path: Path = DEFAULT_PATH, rumble_only: bool = False, random_state: int = None, sample_rate: int = 2000) -> tuple:
    prefetched_files = __prefetch(path)
    metadata = __load_metadata(prefetched_files, rumble_only, random_state, sample_rate)

    # process the data
    print('extracting features')
    clips = [None] * len(metadata)
    data_frames = []
    for stem in tqdm(prefetched_files.keys()):
        # load in audio file
        audio_file = prefetched_files[stem]['audio_path']
        audio, _ = librosa.load(audio_file, sr=sample_rate)

        # extract features
        df = metadata[metadata.file_stem == stem]
        rows = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            row['idx'] = df.iloc[i].name
            clip, features = __extract_clip_features(audio, sample_rate, row)
            if features == None:
                continue
            clips[row['idx']] = clip
            rows.append(features)
        data_frames.append(pd.DataFrame(rows))
    print('features extracted')

    # create dataframe and restore to original ordering
    data = pd.DataFrame(rows).sort_values(by='idx', ignore_index=True)
    return clips, data.drop(columns=['idx'])


def iterclip(path: Path = DEFAULT_PATH, rumble_only: bool = False, random_state: int = None, sample_rate: int = 2000):
    random.seed(random_state)
    prefetched_files = __prefetch(path)
    metadata = __load_metadata(prefetched_files, rumble_only, random_state, sample_rate)

    # lazy load each audio clip
    for i in range(len(metadata)):
        row = metadata.iloc[i].to_dict()

        # load the audio file
        file_stem = row['file_stem']
        audio_file = prefetched_files[file_stem]['audio_path']
        audio, _ = librosa.load(audio_file, sr=sample_rate)
    
        # extract features
        clip, features = __extract_clip_featues(audio, sample_rate, row)
        if features == None:
            continue
        
        yield clip, features

def __prefetch(path: Path) -> dict:
    # check to see if data is already on local machine and download if not.
    annotations_path, audio_path = path.joinpath('manualAnnotations'), path.joinpath('audiomoth')
    if not __validate_data_path(path): 
        a3em.utils.clean_directory(path)
        __download_data(path)

    # format response
    res = {}
    for file in sorted(annotations_path.glob('*txt')):
        res[file.stem] = { 'annotation_path': file }
    for file in sorted(audio_path.glob('*.WAV')):
        res[file.stem]['audio_path'] = file
    
    print('prefetch complete')
    return res


def __load_metadata(prefetched_files: dict, rumble_only: bool, random_state: int, sample_rate: int) -> pd.DataFrame:
    metadata = pd.DataFrame()
    for file_stem in prefetched_files.keys():
        annotation_file = prefetched_files[file_stem]['annotation_path']
        annotations = pd.read_csv(annotation_file, sep='\t')
        file_metadata = __extract_rumbles(annotations, sample_rate, file_stem)
        if len(file_metadata) == 0:
            continue
        if not rumble_only:
            background_noise_metadata = __extract_background_noise(file_metadata, sample_rate, file_stem)
            file_metadata = pd.concat([file_metadata, background_noise_metadata], ignore_index=True)
        file_metadata = __filter_rumbles(file_metadata)
        metadata = file_metadata if metadata.empty else pd.concat([metadata, file_metadata], ignore_index=True)
    return metadata.sample(frac=1, random_state=random_state).reset_index(drop=True)


def __validate_data_path(path: Path) -> bool:
    print('checking for existing files')
    annotations_path, audio_path = path.joinpath('manualAnnotations'), path.joinpath('audiomoth')
    if not (annotations_path.is_dir() and audio_path.is_dir()):
        print('local data missing or incomplete')
        return False
    
    # TODO - add more validation

    return True


def __extract_clip_features(audio: np.ndarray, sample_rate: int, row: dict) -> tuple:
    start_sample = max(0, int((row['Begin Time (s)'] - BUFFER)) * sample_rate)
    end_sample = int((row['End Time (s)'] + BUFFER) * sample_rate)
    clip = audio[start_sample:end_sample]
    if len(clip) < sample_rate * 2:
        return None, None
    
    clip_processed = a3em.utils.preprocess(clip, sample_rate)
    features = a3em.utils.extract_features(clip_processed, sample_rate)
    
    return clip, {**row, **features}


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


def __extract_rumbles(annotations: pd.DataFrame, sample_rate: int, file_stem: str) -> pd.DataFrame:
    rows, drop = [], ['Selection', 'View', 'Channel', 'Low Freq (Hz)', 'High Freq (Hz)']
    for i in range(len(annotations)):
        row = annotations.iloc[i].to_dict()
        row['file_stem'] = file_stem
        row['sample_rate'] = sample_rate
        for key in drop:
            del row[key]
        rows.append(row)
    return pd.DataFrame(rows)


def __extract_background_noise(rumble_annotations: pd.DataFrame, sample_rate: int, file_stem: str) -> pd.DataFrame:
    rows = []

    # get ranges for all possible rumbles
    rumble_event_ranges = list(zip(rumble_annotations['Begin Time (s)'], rumble_annotations['End Time (s)']))

    # get average duration of rumbles
    durations = list(map(lambda time: time[1] - time[0], rumble_event_ranges))
    duration_stats = (np.mean(durations), np.std(durations))

    # generate background noise clips from regions confirmed to not have rumbles 
    background_noise_regions = __generate_background_noise_regions(rumble_event_ranges)
    clip_ranges = list(map(lambda x: __generate_random_clip_range(x, duration_stats), background_noise_regions))
   
    # generate rows for data frame
    for clip_start, clip_end in clip_ranges:
        rows.append({
            'call_type': 'BKG',
            'Begin Time (s)': clip_start,
            'End Time (s)': clip_end,
            'quality': 0,
            'overlap': 'N',
            'earflap': 0,
            'file_stem': file_stem,
            'sample_rate': sample_rate,
        })

    return pd.DataFrame(rows)


def __generate_background_noise_regions(rumble_ranges: list) -> list:
    start_times = [0.0]
    end_times = []
    for rumble_start, rumble_end in rumble_ranges:
        start_times.append(rumble_end)
        end_times.append(rumble_start)
    start_times.pop()
    return list(zip(start_times, end_times))


def __generate_random_clip_range(boundary: tuple, duration_stats: tuple) -> tuple:
    start_boundary, end_boundary = boundary
    duration_mean, duration_standard_deviation = duration_stats
        
    center = (end_boundary - start_boundary) * random.random() + start_boundary
    clip_length = duration_mean + (duration_standard_deviation * random.random() * random.randrange(-1, 2, 2))    
    clip_start = max(start_boundary, center - clip_length // 2)
    clip_end = min(center + clip_length // 2, end_boundary)
    
    return clip_start, clip_end

    
def __filter_rumbles(df: pd.DataFrame) -> pd.DataFrame:
    df['earflap'] = pd.to_numeric(df['earflap'], errors='coerce')
    return df[
        (df['call_type'].isin(['RUM', 'BKG'])) &
        (df['earflap'].isin([0, 1])) &
        (df['overlap']  == 'N') & # TODO - change from 'N' to 0
        (df['quality'].isin([0, 2, 3, 4]))
    ]
