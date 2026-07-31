import a3em.utils
import librosa
import random
import requests
import zipfile
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
    
class Dataset(ABC):

    def __init__(self, path, token=None):
        self.path = Path(path)
        self.token = token

    @abstractmethod
    def load_data(self, test_split, random_state, shuffle):
        return None

    @abstractmethod
    def __iter__(self):
        return None

    @abstractmethod
    def __next__(self):
        return None

    @abstractmethod
    def __getitem__(self, key):
        return None

    @abstractmethod
    def __len__(self):
        return None
 
class Arden(Dataset):

    api = 'https://datadryad.org/'
    doi = 'doi%3A10.5061%2Fdryad.xd2547dz3'

    def __init__(self, path, token):
        super().__init__(path, token)
        self.prefetch = None
        self.metadata = None
        self.audiomoth_path = self.path.joinpath('audiomoth')
        self.annotations_path = self.path.joinpath('manualAnnotations')
        self._features = None
        self._clips = None

    # TODO
    def load_data(
        self, 
        test_split=0.2, 
        random_state=None, 
        sample_rate=2000,
        rumble_only=False, 
        reload=False,
        shuffle=False
    ):        
        if self._clips is None or self._features is None or reload:
            self.__prefetch()
            self.__load_metadata(random_state, rumble_only)
            self.__load_audio_features(random_state, sample_rate)

        labels = self.metadata['quality'].replace({0: 0, 2: 1, 3: 1, 4: 1})
        labels.name = 'labels'

        return train_test_split(
            self._features,
            labels,
            test_size=test_split,
            random_state=random_state,
            shuffle=shuffle
        )

    def load_clips(
        self, 
        random_state=None, 
        sample_rate=2000, 
        rumble_only=False,
        reload=False
    ):
        if self._clips is None or self._features is None or reload:
            self.__prefetch()
            self.__load_metadata(random_state, rumble_only)
            self.__load_audio_features(random_state, sample_rate)
        return self._clips, self._features
        

    # TODO    
    def __iter__(self):
        return None

    # TODO
    def __next__(self):
        return None

    def __getitem__(self, key):
        if self._clips is None or self._features is None:
            return None, None
        return self._clips[key], self._features.iloc[key].to_dict()

    def __len__(self):
        return 0 if self.metadata is None else len(self.metadata)
    
    def __prefetch(self):
        if not self.__validate_local_data():
            self.__download_data()
            
        audiomoth_files = sorted(self.audiomoth_path.glob('*.WAV'))
        annotation_files = sorted(self.annotations_path.glob('*.txt'))
        
        file_stems = list(map(lambda x: x.stem, audiomoth_files))
        file_pairs = list(map(
            lambda x: {'audio_path': x[0], 'annotation_path': x[1]},
            zip(audiomoth_files, annotation_files)
        ))
        
        self.prefetch = dict(zip(file_stems, file_pairs))      
        print('prefetch complete')

    def __validate_local_data(self):
        if not (self.audiomoth_path.exists() and self.annotations_path.exists()):
            return False
        audiomoth_contents = sorted(self.audiomoth_path.glob('*.WAV'))
        annotations_contents = sorted(self.annotations_path.glob('*.txt'))
        annotations_stems = list(map(lambda x: x.stem, annotations_contents))
        audiomoth_stems = list(map(lambda x: x.stem, audiomoth_contents))
        return annotations_stems == audiomoth_stems

    # TODO
    def __download_data(self):
        pass

    def __load_metadata(self, random_state, rumble_only):
        metadata = pd.DataFrame()
        for stem in self.prefetch.keys():
            annotation_path = self.prefetch[stem]['annotation_path']
            annotations = pd.read_csv(annotation_path, sep='\t')
            if annotations.empty:
                continue

            # extract elephant rumbles
            file_metadata = Arden.__filter_annotations(annotations, stem)

            # extract background noise if applicable
            if not rumble_only:
                noise_metadata = Arden.__extract_noise(file_metadata, stem)
                file_metadata = pd.concat(
                    [file_metadata, noise_metadata], 
                    ignore_index=True
                )

            # filter rumbles and add to dataframe
            file_metadata = Arden.__filter_clips(file_metadata)
            metadata = pd.concat([metadata, file_metadata], ignore_index=True)

        self.metadata = (
            metadata
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )

    def __load_audio_features(self, random_state=None, sample_rate=2000):
        random.seed(random_state)
        clips = [None] * len(self)
        features = [None] * len(self)
        
        print('extracting features')
        for stem in tqdm(self.prefetch.keys()):
            # load in audio file
            audio_file = self.prefetch[stem]['audio_path']
            audio, _ = librosa.load(audio_file, sr=sample_rate)

            df = self.metadata[self.metadata.file_stem == stem]
            for index, row in df.iterrows():
                start_time = row['Begin Time (s)']
                end_time = row['End Time (s)']

                clip, feature_set = Arden.__extract_clip_features(
                    audio, 
                    sample_rate, 
                    start_time, 
                    end_time
                )

                clips[index] = clip 
                features[index] = feature_set
        
        self._features = pd.DataFrame(features)
        self._clips = clips

    @staticmethod
    def __filter_annotations(annotations, stem):
        d = ['Selection', 'View', 'Channel', 'Low Freq (Hz)', 'High Freq (Hz)']
        df = annotations.drop(columns=d)
        df.insert(0, 'file_stem', [stem] * len(annotations))
        return df
        
    @staticmethod
    def __extract_noise(rumble_annotations, stem):
        # get ranges for all possible rumbles
        rumble_event_ranges = list(zip(
            rumble_annotations['Begin Time (s)'],
            rumble_annotations['End Time (s)']
        ))

        # get average duration of rumbles
        rumble_deltas = list(map(lambda t: t[1] - t[0], rumble_event_ranges))
        rumble_delta_stats = np.mean(rumble_deltas), np.std(rumble_deltas)

        # generate noise clips from regions without rumbles
        noise_regions = Arden.__find_noise_regions(rumble_event_ranges)
        clip_ranges = list(map(
            lambda x: Arden.__random_clip_range(x, rumble_delta_stats),
            noise_regions
        ))

        # create dataframe
        rows = []
        for clip_start, clip_end in clip_ranges:
            rows.append({
                'file_stem': stem,
                'call_type': 'BKG',
                'Begin Time (s)': clip_start,
                'End Time (s)': clip_end,
                'quality': 0,
                'overlap': 'N',
                'earflap': 0
            })

        return pd.DataFrame(rows)

    @staticmethod
    # FIXME filter for clip duration as well
    def __filter_clips(df):
        df['earflap'] = pd.to_numeric(df['earflap'], errors='coerce')
        return df[
            (df['call_type'].isin(['RUM', 'BKG']))
            & (df['earflap'].isin([0, 1]))
            & (df['overlap'] == 'N')
            & (df['quality'].isin([0, 2, 3, 4]))
        ]

    @staticmethod
    def __find_noise_regions(rumble_ranges):
        start_times = [x[0] for x in rumble_ranges]
        end_times = [x[1] for x in rumble_ranges]
        start_times.pop()
        start_times.insert(0, 0.0)
        return list(zip(start_times, end_times))
        
    @staticmethod
    def __random_clip_range(boundary, delta_stats):
        start_bound, end_bound = boundary
        delta_mean, delta_std = delta_stats

        center = (end_bound - start_bound) * random.random() + start_bound
        clip_length = (
            delta_mean 
            + delta_std 
            * random.random() 
            * random.randrange(-1, 2, 2)
        )

        clip_start = max(start_bound, center - clip_length / 2)
        clip_end = min(center + clip_length / 2, end_bound)

        return clip_start, clip_end

    @staticmethod
    def __extract_clip_features(audio, sample_rate, start_time, end_time):
        buffer = 0.2
        start_sample = max(0, int((start_time - buffer) * sample_rate))
        end_sample = int((end_time + buffer) * sample_rate)

        clip = audio[start_sample:end_sample]
        if len(clip) < sample_rate * 2:
            return [], {}

        preprocessed_clip = a3em.utils.preprocess(clip, sample_rate)
        features = a3em.utils.extract_features(preprocessed_clip, sample_rate)

        return clip, features
