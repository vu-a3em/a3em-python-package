import a3em.utils
from datetime import datetime
import librosa
import numpy as np
import os
import pandas as pd
from pathlib import Path

DEFAULT_PREFETCH_PATH = Path('./a3em/datasets/arden_data')
# TODO - add default audio path (pull data from db?)


def load_data(test_split: float, seed: int, prefetch_path: Path = None) -> tuple:
    if test_split < 0.0 or test_split > 1.0:
        raise ValueError('the test split fraction must be between 0.0 and 1.0')

    # pre-load audio
    audio_metadata = __prefetch(prefetch_path)
    __validate_prefetch(audio_metadata)

    # create dataframe

    # create split

    return


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
        audio, sample_rate = librosa.load(file_path)
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
        

def __generate_dataframe(audio_metadata: pd.DataFrame) -> pd.DataFrame:
    annotation_directory = Path(os.getenv('ANNOTATION_PATH'))
    annotation_files = sorted(annotation_directory.glob('*.txt'))

    all_rows = []
    for annotation_path in annotation_files:
        quality_rumble_annotations: pd.DataFrame = __isolate_high_quality_rumbles(annotation_path)
        annotation_path_stem: str = annotation_path.stem
        recording_start: datetime = __parse_start_time(annotation_path)

        # pull audio data
        # audio, sample_rate = 

    return all_rows


def __isolate_high_quality_rumbles(annotation_path: Path) -> pd.DataFrame:
    df = pd.read_csv(annotation_path, sep='\t')
    df['earflap'] = pd.to_numeric(df['earflap'], errors='coerce')
    df = df[
        (df['call_type'] == 'RUM') &
        (df['earflap'].isin([0])) &
        (df['quality'].isin([3, 4])) &
        (df['overlap']  == 'N')
    ]
    return df


def __parse_start_time(annotation_path_stem: str) -> datetime:
    parts = annotation_path_stem.split('_')
    return datetime.strptime(parts[1] + parts[2], '%Y%m%d%H%M%S')


# def __load_audio(annotation_path, audio_aggregrate)