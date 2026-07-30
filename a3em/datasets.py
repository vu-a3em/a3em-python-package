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

    # TODO
    def load_data(
        self, 
        test_split=0.2, 
        random_state=None, 
        sample_rate=2000,
        rumble_only=False, 
        shuffle=False
    ):
        self.__prefetch()
        self.__load_metadata(random_state, sample_rate, rumble_only)
        return None    

    def load_clips(
        self, 
        random_state=None, 
        sample_rate=2000, 
        rumble_only=False
    ):
        self.__prefetch()
        self.__load_metadata(random_state, sample_rate, rumble_only)

        # process the data
        print('extracting features')
        

    # TODO    
    def __iter__(self):
        return None

    # TODO
    def __next__(self):
        return None

    # TODO
    def __getitem__(self, key):
        return None

    # TODO
    def __len__(self):
        return None
    
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

    # FIXME - can this be updated with a try catch?
    def __validate_local_data(self):
        # check audiomoth
        audiomoth = self.path.joinpath('audiomoth')
        if not audiomoth.exists():
            return False
        audiomoth_contents = sorted(audiomoth.glob('*.WAV'))
        if len(audiomoth_contents) != 62:
            return False
        
        # check manualAnnotations
        annotations = self.path.joinpath('manualAnnotations')
        if not annotations.exists():
            return False
        annotations_contents = sorted(annotations.glob('*.txt'))
        if len(annotations_contents) != 62:
            return False
                
        # check that each annotation file has an audio file 
        annotations_stems = list(map(lambda x: x.stem, annotations_contents))
        audiomoth_stems = list(map(lambda x: x.stem, audiomoth_contents))
        return annotations_stems == audiomoth_stems

    # TODO
    def __download_data(self):
        pass

    def __load_metadata(self, random_state, sample_rate, rumble_only):
        metadata = pd.DataFrame()
        for stem in self.prefetch.keys():
            annotation_path = self.prefetch[stem]['annotation_path']
            annotations = pd.read_csv(annotation_path, sep='\t')
            if annotations.empty:
                continue

            # extract elephant rumbles
            file_metadata = Arden.__filter_annotations(
                annotations, 
                sample_rate, 
                stem
            )

            # extract background noise if applicable
            if not rumble_only:
                noise_metadata = Arden.__extract_noise(
                    file_metadata, 
                    sample_rate, 
                    stem
                )
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

    @staticmethod
    def __filter_annotations(annotations, sample_rate, stem):
        d = ['Selection', 'View', 'Channel', 'Low Freq (Hz)', 'High Freq (Hz)']
        df = annotations.drop(columns=d)
        df.insert(0, 'sample_rate', [sample_rate] * len(annotations))
        df.insert(0, 'file_stem', [stem] * len(annotations))
        return df
        
    @staticmethod
    def __extract_noise(rumble_annotations, sample_rate, stem):
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
                'sample_rate': sample_rate,
                'call_type': 'BKG',
                'Begin Time (s)': clip_start,
                'End Time (s)': clip_end,
                'quality': 0,
                'overlap': 'N',
                'earflap': 0
            })

        return pd.DataFrame(rows)

    @staticmethod
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
    
