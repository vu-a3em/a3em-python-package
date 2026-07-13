import a3em.utils
from datetime import datetime, timedelta
import librosa
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BUFFER = 0.2
DEFAULT_PREFETCH_PATH = Path('./a3em/datasets/arden_data')
# TODO - add default audio path (pull data from db?)


def load_data(
        prefetch_path: Path = None, quality_check: bool = True, test_split: float = 0.25, 
        random_state: int = None, shuffle: bool = True) -> tuple:
    if test_split < 0.0 or test_split > 1.0:
        raise ValueError('the test split fraction must be between 0.0 and 1.0')
    
    rumbles = get_quality_rumbles(prefetch_path) if quality_check else get_all_rumbles(prefetch_path)

    return False
    #return train_test_split(df, test_size=test_split, random_state=random_state, shuffle=shuffle)


def get_quality_rumbles(prefetch_path: Path = None) -> pd.DataFrame:
    audio_metadata = __prefetch(prefetch_path)
    _ = __validate_prefetch(audio_metadata)
    df = __generate_dataframe(audio_metadata)
    return df


def get_all_rumbles(prefetch_path: Path = None) -> pd.DataFrame:
    audio_metadata = __prefetch(prefetch_path)
    _ = __validate_prefetch(audio_metadata)
    df = __generate_dataframe(audio_metadata, quality_check=False)
    return df


# TODO - display a status bar
def __prefetch(prefectch_path: Path) -> pd.DataFrame:
    # TODO - give more robust checks. we should make sure the dataset is complete
    prefectch_path = DEFAULT_PREFETCH_PATH if prefectch_path == None else prefectch_path
    metadata_path = os.path.join(prefectch_path, 'metadata.csv')

    contents = os.listdir(prefectch_path)
    if 'metadata.csv' in contents:
        df = pd.read_csv(metadata_path, index_col='Unnamed: 0')
        return df
    
    audio_directory = Path(os.getenv('AUDIO_PATH'))
    audio_files = sorted(audio_directory.glob('*.wav'))
    names, paths, sample_rates = [], [], []

    for file in audio_files:
        file_path = os.path.join(audio_directory, file)
        audio, sample_rate = librosa.load(file_path, sr=2000)
        data_path = os.path.join(prefectch_path, file.stem + '.npy')
        np.save(data_path, audio)

        names.append(file.stem)
        paths.append(data_path)
        sample_rates.append(sample_rate)

    df = pd.DataFrame({'path': paths, 'sample_rate': sample_rates}, index=names)
    df.to_csv(metadata_path)
    return df 


def __validate_prefetch(audio_metadata: pd.DataFrame) -> bool:
    indecies = audio_metadata.index
    for i in indecies:
        path = Path(audio_metadata.loc[i].path)
        if path.stem != i:
            raise IndexError('metadata not properly indexed')
        if not path.is_file():
            raise RuntimeError(f'file not found: {path}')
    return True
        

def __generate_dataframe(audio_metadata: pd.DataFrame, quality_check: bool = True) -> pd.DataFrame:
    annotation_directory = Path(os.getenv('ANNOTATION_PATH'))
    annotation_files = sorted(annotation_directory.glob('*.txt'))

    all_rows = []
    for annotation_path in annotation_files:
        annotation_path_stem: str = annotation_path.stem
        recording_start: datetime = __parse_start_time(annotation_path_stem)

        audio_path, sample_rate = audio_metadata.loc[annotation_path_stem]
        audio = np.load(audio_path)

        rumble_annotations: pd.DataFrame = __isolate_high_quality_rumbles(annotation_path) if quality_check else __isolate_rumbles(annotation_path)
        for i in range(len(rumble_annotations)):
            row = rumble_annotations.iloc[i]

            start_sample = int((row['Begin Time (s)'] - BUFFER) * sample_rate)
            end_sample = int((row['End Time (s)'] + BUFFER) * sample_rate)
            start_time = recording_start + timedelta(seconds = row['Begin Time (s)'])
            start_time.strftime('%Y%m%d_%H%M%S')
            
            clip = audio[start_sample:end_sample]
            clip_processed = a3em.utils.preprocess(clip, sample_rate)
            features = a3em.utils.extract_features(clip_processed, sample_rate)

            combined = {
                'filename': annotation_path_stem,
                'rec_start': recording_start,
                'abs_begin': recording_start + timedelta(seconds=row['Begin Time (s)']),
                'abs_end': recording_start + timedelta(seconds=row['End Time (s)']),
                'duration': row['End Time (s)'] - row['Begin Time (s)'],
                'quality': row['quality'],
                **row.to_dict(),
                **features,
            }

            all_rows.append(combined)

    return pd.DataFrame(all_rows)


def __isolate_rumbles(annotation_path: Path) -> pd.DataFrame:
    df = pd.read_csv(annotation_path, sep='\t')
    df['earflap'] = pd.to_numeric(df['earflap'], errors='coerce')
    df = df[
        (df['call_type'] == 'RUM') &
        (df['earflap'].isin([0])) &
        (df['overlap']  == 'N')
    ]
    return df


def __isolate_high_quality_rumbles(annotation_path: Path) -> pd.DataFrame:
    df = __isolate_rumbles(annotation_path)
    df = df[df['quality'].isin([3, 4])]
    return df


def __parse_start_time(annotation_path_stem: str) -> datetime:
    parts = annotation_path_stem.split('_')
    return datetime.strptime(parts[1] + parts[2], '%Y%m%d%H%M%S')
