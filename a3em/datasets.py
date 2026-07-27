import a3em.utils
import librosa
import os
import random
import requests
import zipfile
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

class Arden:

    _api: str = 'https://datadryad.org/'
    _buffer = 0.2
    _default_data_path: str = './data'
    _doi: str = 'doi%3A10.5061%2Fdryad.xd2547dz3'
    
    # TODO
    @staticmethod
    def load_data(
            api_token: str, path: str, test_split: float = 0.2, 
            random_state: int = None, shuffle: bool = False) -> tuple:
        data_prefetch = Arden.__prefetch(api_token, path)
        return data_prefetch

    @staticmethod
    def __prefetch(api_token: str, path: str) -> dict:
        path = Path(path)
        if not Arden.__validate_local_data(path):
            Arden.__download_data(api_token, path)            
        
        # format response
        res = {}
        for file in sorted(path.joinpath('manualAnnotations').glob('*.txt')):
            res[file.stem] = {'annotation_path': file}
        for file in sorted(path.joinpath('audiomoth').glob('*.WAV')):
            res[file.stem]['audio_path'] = file
            
        print('prefetch complete')
        return res

    # TODO
    @staticmethod
    def __validate_local_data(path: str) -> bool:
        return False

    @staticmethod
    def iter(
            api_token: str, path: str, rumble_only: bool = False, 
            random_state: int = None, sample_rate: int = 2000) -> tuple:
        random.seed(random_state)
        prefetch = Arden.__prefetch(api_token, path)
        metadata = Arden.__load_metadata(
            prefetch, rumble_only, random_state, sample_rate)

        # lazy load each audio clip
        for i in range(len(metadata)):
            row = metadata.iloc[i].to_dict()

            # load the audio file
            stem = row['file_stem']
            audio_file = prefetch[stem]['audio_path']
            audio, _ = librosa.load(audio_file, sr=sample_rate)

            # extract features
            clip, features = Arden.__extract_clip_features(
                audio, sample_rate, row)
            if not features:
                continue

            yield clip, features

    @staticmethod
    def __download_data(api_token: str, path: str):      
        # make sure the directory is clear
        a3em.utils.clean_directory(path)

        # pull files metadata
        print('pulling metadata')
        r = requests.get(f'{Arden._api}/api/v2/datasets/{Arden._doi}/versions')
        if r.status_code != 200:
            raise RuntimeError(r.text)
        content = r.json()
        latest_version = content['_embedded']['stash:versions'][0]
        files_path = latest_version['_links']['stash:files']['href']

        # get individual download links
        print('locating files')
        r = requests.get(f'{Arden._api}/{files_path}')
        if r.status_code != 200:
            raise RuntimeError(r.text)
        content = r.json()
        files = content['_embedded']['stash:files']

        # download the files
        print('download in progress')
        for file in files:
            download_src = file['_links']['stash:download']['href']
            download_dst = path.joinpath(file['path'])
            r = requests.get(f'{Arden._api}/{download_src}',
                             headers={'authorization': f'Bearer {api_token}'})
            if r.status_code != 200:
                raise RuntimeError(r.text)
            with open(download_dst, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=128):
                    fd.write(chunk)

        # unpack the files
        print('unpacking')
        zip_files = sorted(path.glob('*.zip'))
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(path)
            zip_file.unlink()

    @staticmethod
    def __load_metadata(
            prefetch: dict, rumble_only: bool, 
            random_state: int, sample_rate: int) -> pd.DataFrame:
        metadata = pd.DataFrame()
        for stem in prefetch.keys():
            annotation_path = prefetch[stem]['annotation_path']
            annotations = pd.read_csv(annotation_path, sep='\t')
            file_metadata = Arden.__extract_rumbles(
                annotations, sample_rate, stem)
            if len(file_metadata) == 0:
                continue
            if not rumble_only:
                background_noise_metadata = Arden.__extract_background_noise(
                    file_metadata, sample_rate, stem)
                file_metadata = pd.concat(
                    [file_metadata, background_noise_metadata], 
                    ignore_index=True)
            file_metadata = Arden.__filter_rumbles(file_metadata)
            metadata = file_metadata if metadata.empty else pd.concat(
                [metadata, file_metadata], ignore_index=True)
        return metadata.sample(frac=1, random_state=random_state).reset_index(
            drop=True)
    
    @staticmethod
    def __extract_clip_features(
                audio: np.ndarray, sample_rate: int, row: dict) -> tuple:
        start_sample = max(
            0, int((row['Begin Time (s)'] - Arden._buffer)) * sample_rate)
        end_sample = int((row['End Time (s)'] + Arden._buffer) * sample_rate)
        clip = audio[start_sample:end_sample]
        if len(clip) < sample_rate * 2:
            return None, None
        
        clip_processed = a3em.utils.preprocess(clip, sample_rate)
        features = a3em.utils.extract_features(clip_processed, sample_rate)
        
        return clip, {**row, **features}
    
    @staticmethod
    def __extract_rumbles(
            annotations: pd.DataFrame, sample_rate: int, 
            file_stem: str) -> pd.DataFrame:
        rows = []
        drop = [
            'Selection', 'View', 'Channel', 
            'Low Freq (Hz)', 'High Freq (Hz)'
        ]
        
        for i in range(len(annotations)):
            row = annotations.iloc[i].to_dict()
            row['file_stem'] = file_stem
            row['sample_rate'] = sample_rate
            for key in drop:
                del row[key]
            rows.append(row)
            
        return pd.DataFrame(rows)
    
    @staticmethod
    def __extract_background_noise(
            rumble_annotations: pd.DataFrame, sample_rate: int, 
            file_stem: str) -> pd.DataFrame:
        rows = []

        # get ranges for all possible rumbles
        rumble_event_ranges = list(zip(
            rumble_annotations['Begin Time (s)'], 
            rumble_annotations['End Time (s)']
        ))

        # get average duration of rumbles
        durations = list(map(
            lambda time: time[1] - time[0], rumble_event_ranges))
        duration_stats = (np.mean(durations), np.std(durations))

        # generate background noise clips from regions confirmed to 
        # not have rumbles 
        background_noise_regions = Arden.__generate_background_noise_regions(
            rumble_event_ranges)
        clip_ranges = list(map(
            lambda x: Arden.__generate_random_clip_range(x, duration_stats),
            background_noise_regions
        ))
    
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
    
    @staticmethod
    def __generate_background_noise_regions(rumble_ranges: list) -> list:
        start_times = [0.0]
        end_times = []
        for rumble_start, rumble_end in rumble_ranges:
            start_times.append(rumble_end)
            end_times.append(rumble_start)
        start_times.pop()
        return list(zip(start_times, end_times))
    
    @staticmethod
    def __generate_random_clip_range(
        boundary: tuple, duration_stats: tuple) -> tuple:
        start_boundary, end_boundary = boundary
        duration_mean, duration_standard_deviation = duration_stats
            
        center = ((end_boundary - start_boundary) 
            * random.random() + start_boundary)
        clip_length = (duration_mean 
            + (duration_standard_deviation * random.random() 
            * random.randrange(-1, 2, 2)))    
        clip_start = max(start_boundary, center - clip_length // 2)
        clip_end = min(center + clip_length // 2, end_boundary)
        
        return clip_start, clip_end
    
    @staticmethod
    def __filter_rumbles(df: pd.DataFrame) -> pd.DataFrame:
        df['earflap'] = pd.to_numeric(df['earflap'], errors='coerce')
        return df[
            (df['call_type'].isin(['RUM', 'BKG'])) &
            (df['earflap'].isin([0, 1])) &
            (df['overlap']  == 'N') & # TODO - change from 'N' to 0
            (df['quality'].isin([0, 2, 3, 4]))
        ]

class Dataset(ABC):

    @staticmethod
    @abstractmethod
    def load_data(
            path: str, test_split: float, 
            random_state: int, shuffle: bool) -> tuple:
        return None,

    @staticmethod
    @abstractmethod
    def __prefetch(path: str) -> dict:
        return {}

    @staticmethod
    @abstractmethod
    def __validate_local_data(path: str) -> bool:
        return False

    @staticmethod
    @abstractmethod
    def iter() -> tuple:
        return None,

Dataset.register(Arden)
