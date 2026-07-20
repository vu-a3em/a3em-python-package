import a3em.utils
import librosa
import os
import pandas as pd
import requests
import zipfile
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

BUFFER = 0.2
DEFAULT_PATH='./data/'

def load_data(
        path: Path = DEFAULT_PATH, 
        sample_rate: int = None, 
        test_split: float = 0.25,
        random_state: int = 123,
        shuffle: bool = True
) -> tuple:
    _, df = load_clips(path, sample_rate=sample_rate)
    labels = df['quality'].replace({
        0: 0,
        3: 1,
        4: 1
    })
    labels.name = 'label'
    data = df.drop(columns='quality')
    if test_split == 0.0:
        return data, labels
    return train_test_split(data, labels, test_size=test_split, random_state=random_state, shuffle=shuffle)
    

def load_clips(
        path: Path = DEFAULT_PATH, 
        rumble_only: bool = False, 
        noise_seed: int = None, 
        sample_rate: int = 2000
) -> tuple:
    print('loading audio data')
    audio_files, annotation_files = __prefetch(path)
    audio_dict = {}
    for file in audio_files: 
        audio_dict[file.stem] = file  
   
    print('extracting rumbles')
    clip_tuples = []
    for file in annotation_files:
        audio, sr = librosa.load(audio_dict[file.stem], sr=sample_rate)
        rumble_annotations = __filter_rumbles(file)
        for i in range(len(rumble_annotations)):
            row = rumble_annotations.iloc[i]
            start_sample = max(0, int((row['Begin Time (s)'] - BUFFER) * sr))
            end_sample = min(len(audio), int((row['End Time (s)'] + BUFFER) * sr))
            clip = audio[start_sample:end_sample]
            clip_tuples.append((clip, row))
    
    # TODO - extract background noise
        
    # format the clip data
    clips, rows = [], []
    for clip, row in clip_tuples:
        clips.append(clip)
        rows.append(row)

    print('extraction complete')
    return clips, pd.DataFrame(rows)


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


def __filter_rumbles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    df['earflap'] = pd.to_numeric(df['earflap'], errors='coerce')
    return df[
        (df['call_type'] == 'RUM') &
        (df['earflap'].isin([0, 1])) &
        (df['overlap']  == 'N') & # TODO - change from 'N' to 0
        (df['quality'].isin([2, 3, 4]))
    ]


def __parse_start_time(annotation_path: Path) -> datetime:
    parts = annotation_path.stem.split('_')
    return datetime.strptime(parts[1] + parts[2], '%Y%m%d%H%M%S')