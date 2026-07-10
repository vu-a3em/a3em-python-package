import a3em.utils
from datetime import datetime
import librosa
import numpy as np
import os
import pandas as pd


def load_data(test_split: float, seed: int) -> tuple:
    if test_split < 0.0 or test_split > 1.0:
        raise ValueError('the test split fraction must be between 0.0 and 1.0')

    # pre-load audio

    # create dataframe

    # create split

    return


# TODO - figure out a way to cache this data
# TODO - display a status bar
# TODO - make this a dict
def __prefetch() -> list:
    audio_directory = os.getenv('AUDIO_PATH')
    audio_files = os.listdir(audio_directory)
    audio_aggregate = []
    for file in audio_files:
        path = os.path.join(audio_directory, file)
        audio, sample_rate = librosa.load(path)
        data = { 'audio': audio, 'sample_rate': sample_rate, 'file_name': file }
        audio_aggregate.append(data)
    return audio_aggregate
        

def __generate_dataframe(audio_aggregate: list) -> pd.DataFrame:
    annotation_directory = os.getenv('ANNOTATION_PATH')
    annotation_files = os.listdir(annotation_directory)

    all_rows = []
    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotation_directory, annotation_file)
        quality_rumble_annotations: pd.DataFrame = __isolate_high_quality_rumbles(annotation_path)
        recording_start: datetime = __parse_start_time(annotation_path)

        # # pull audio data
        print(audio_aggregate)
        #audio_metadata = audio_aggregate[f'{ann]
        # audio, sample_rate = 

    return all_rows


def __isolate_high_quality_rumbles(annotation_path: str) -> pd.DataFrame:
    df = pd.read_csv(annotation_path, sep='\t')
    df['earflap'] = pd.to_numeric(df['earflap'], errors='coerce')
    df = df[
        (df['call_type'] == 'RUM') &
        (df['earflap'].isin([0])) &
        (df['quality'].isin([3, 4])) &
        (df['overlap']  == 'N')
    ]
    return df


def __parse_start_time(annotation_path: str) -> datetime:
    stem = a3em.utils.__get_path_stem(annotation_path)
    parts = stem.split('_')
    return datetime.strptime(parts[1] + parts[2], '%Y%m%d%H%M%S')