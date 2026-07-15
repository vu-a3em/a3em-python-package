import a3em.utils
from datetime import datetime, timedelta
import librosa
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

BUFFER = 0.2
DEFAULT_PREFETCH_PATH = Path('./a3em/datasets/arden_data')
DROP_COLUMNS = [
    'filename', 'rec_start', 'abs_begin', 'abs_end', 
    'duration', 'Selection', 'View', 'Channel',
    'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)',
    'call_type', 'overlap', 'earflap',
]
# TODO - add default audio path (pull data from db?)


def load_data(
        prefetch_path: Path = None, quality_check: bool = True, 
        test_split: float = 0.25, random_state: int = None, shuffle: bool = True) -> tuple:
    if test_split < 0.0 or test_split > 1.0:
        raise ValueError('the test split fraction must be between 0.0 and 1.0')
    
    audio_metadata = __prefetch(prefetch_path)

    annotation_directory = Path(os.getenv('ANNOTATION_PATH'))
    annotation_files = sorted(annotation_directory.glob('*.txt'))

    rumbles = __generate_rumbles_dataframe(audio_metadata, annotation_files)
    background_noise = __generate_background_noise_dataframe(audio_metadata, annotation_files)

    df = pd.concat([rumbles, background_noise], axis=0).drop(columns=DROP_COLUMNS)
    labels = df['quality']
    data = df.drop(columns='quality')

    return train_test_split(data, labels, test_size=test_split, random_state=random_state, shuffle=shuffle)

# TODO - display a status bar
def __prefetch(prefectch_path: Path = None) -> pd.DataFrame:
    print('prefetching data...')

    # TODO - give more robust checks. we should make sure the dataset is complete
    prefectch_path = DEFAULT_PREFETCH_PATH if prefectch_path == None else prefectch_path
    metadata_path = os.path.join(prefectch_path, 'metadata.csv')

    contents = os.listdir(prefectch_path)
    if 'metadata.csv' in contents:
        print('local data found')
        df = pd.read_csv(metadata_path, index_col='Unnamed: 0')
        _ = __validate_prefetch(df)
        return df
    
    print('loading data...')

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

    print(f'prefetch complete: metadata can be found at {metadata_path}') 

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
        

def __generate_rumbles_dataframe(audio_metadata: pd.DataFrame, annotation_files: list, quality_check: bool = True) -> pd.DataFrame:
    all_rows = []
    for annotation_path in annotation_files:
        recording_start: datetime = __parse_start_time(annotation_path)

        audio_path, sample_rate = audio_metadata.loc[annotation_path.stem]
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
                'filename': annotation_path.stem,
                'rec_start': recording_start,
                'abs_begin': recording_start + timedelta(seconds=row['Begin Time (s)']),
                'abs_end': recording_start + timedelta(seconds=row['End Time (s)']),
                'duration': row['End Time (s)'] - row['Begin Time (s)'],
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


def __generate_background_noise_dataframe(audio_metadata: pd.DataFrame, annotation_files: list, seed: int = 124) -> pd.DataFrame:    
    #return pd.read_csv(annotation_files[0], sep='\t')
    random.seed(seed)
    all_rows = []
    for annotation_path in annotation_files:
        # load audio data
        recording_start: datetime = __parse_start_time(annotation_path)
        audio_path, sample_rate = audio_metadata.loc[annotation_path.stem]
        audio = np.load(audio_path)
        audio_length = len(audio) / sample_rate

        # use all recorded annotations to guarantee no overlap between background noise and event of interest
        annotations = pd.read_csv(annotation_path, sep='\t')
        rumbles = [
            (annotations['call_type'] == 'RUM') &
            (annotations['earflap'].isin([0])) &
            (annotations['overlap']  == 'N') &
            (annotations['quality'].isin([3, 4]))
        ]

        # find the start and end time of each event
        rumble_event_times = annotations[['Begin Time (s)', 'End Time (s)']].to_dict()
        rumble_event_ranges = []
        for i in range(0, len(rumble_event_times['Begin Time (s)'])):
            rumble_event_ranges.append((rumble_event_times['Begin Time (s)'][i], rumble_event_times['End Time (s)'][i]))
        if len(rumble_event_ranges) == 0:
            continue

        # find the average clip duration and choose an appropriate time variance
        durations = list(map(lambda time: time[1] - time[0], rumble_event_ranges))
        duration_mean = np.mean(durations)
        duration_standard_deviation = np.std(durations)

        # generate new time ranges that do not interfere with the event time ranges
        # there should be one clip of background noise for each rumble clip
        clip_count = len(rumbles)
        clip_ranges = []
        while len(clip_ranges) < clip_count:
            center = audio_length * random.random()
            clip_length = duration_mean + (duration_standard_deviation * random.random() * random.randrange(-1, 2, 2))
            clip_range = (center - clip_length * 0.5, center + clip_length * 0.5)
            if __range_is_background_noise(clip_range, rumble_event_ranges):
                clip_ranges.append(clip_range)
        
        # extract audio features from clip
        for clip_start, clip_end in clip_ranges:
            row = {
                'Selection': None, 
                'View': None, 
                'Channel': 1, 
                'Begin Time (s)': clip_start, 
                'End Time (s)': clip_end,
                'Low Freq (Hz)': None,
                'High Freq (Hz)': None,
                'call_type': 'BgN',
                'quality': 0,
                'overlap': 'N',
                'earflap': 0
            }

            start_sample, end_sample = int(clip_start * sample_rate), int(clip_end * sample_rate)
            start_time = recording_start + timedelta(seconds = clip_start)
            start_time.strftime('%Y%m%d_%H%M%S')
            
            clip = audio[start_sample:end_sample]
            if len(clip) == 0:
                continue
            clip_processed = a3em.utils.preprocess(clip, sample_rate)
            features = a3em.utils.extract_features(clip_processed, sample_rate)

            combined = {
                'filename': annotation_path.stem,
                'rec_start': recording_start,
                'abs_begin': recording_start + timedelta(seconds=row['Begin Time (s)']),
                'abs_end': recording_start + timedelta(seconds=row['End Time (s)']),
                'duration': row['End Time (s)'] - row['Begin Time (s)'],
                **row,
                **features,
            }

            all_rows.append(combined)

    return pd.DataFrame(all_rows)
        

def __range_is_background_noise(clip_range: tuple, rumble_ranges: list) -> bool:
    if len(rumble_ranges) == 0:
        return True

    (clip_start, clip_end) = clip_range
    for rumble_start, rumble_end in rumble_ranges:
        if (clip_start > rumble_start and clip_start < rumble_end) or (clip_end > rumble_start and clip_end < rumble_end):
            return False
        
    return True
    

def __parse_start_time(annotation_path: Path) -> datetime:
    parts = annotation_path.stem.split('_')
    return datetime.strptime(parts[1] + parts[2], '%Y%m%d%H%M%S')
