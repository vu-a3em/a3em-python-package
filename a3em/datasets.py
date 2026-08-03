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
        self._index = 0

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
        self.audiomoth_path = self.path / 'audiomoth'
        self.annotations_path = self.path / 'manualAnnotations'
        self._features = None
        self._clips = None

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
        labels.name = 'label'

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

    def __iter__(
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
            
    # FIXME - try to lazy load the data
    def __next__(self):
        clip = self._clips[self._index]
        features = self._features.iloc[self._index].to_dict()
        self._index += 1
        return clip, features

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
        
        file_stems = [file.stem for file in audiomoth_files]
        file_pairs = [
            {'audio_path': x[0], 'annotation_path': x[1]}
            for x in zip(audiomoth_files, annotation_files)
        ]
        
        self.prefetch = dict(zip(file_stems, file_pairs))      
        print('prefetch complete')

    def __validate_local_data(self):
        if not (self.audiomoth_path.exists() and self.annotations_path.exists()):
            return False
        audiomoth_contents = sorted(self.audiomoth_path.glob('*.WAV'))
        annotations_contents = sorted(self.annotations_path.glob('*.txt'))
        annotations_stems = [file.stem for file in annotations_contents]
        audiomoth_stems = [file.stem for file in audiomoth_contents]
        return annotations_stems == audiomoth_stems

    def __download_data(self):
        # make sure the directory is clear
        a3em.utils.clean_directory(self.path)

        # pull files metadata
        print('pulling metadata')
        r = requests.get(f'{Arden.api}/api/v2/datasets/{Arden.doi}/versions')
        if r.status_code != 200:
            raise RuntimeError(r.text)
        content = r.json()
        latest_version = content['_embedded']['stash:versions'][0]
        files_path = latest_version['_links']['stash:files']['href']

        # get individual download links
        print('locating files')
        r = requests.get(f'{Arden.api}/{files_path}')
        if r.status_code != 200:
            raise RuntimeError(r.text)
        content = r.json()
        files = content['_embedded']['stash:files']

        # download the files
        print('download in progress')
        for file in files:
            download_src = file['_links']['stash:download']['href']
            download_dst = self.path / file['path']
            r = requests.get(f'{Arden.api}/{download_src}',
                             headers={'authorization': f'Bearer {self.token}'})
            if r.status_code != 200:
                raise RuntimeError(r.text)
            with open(download_dst, 'wb') as fd:
                for chunk in r.iter_content(chunk_size=128):
                    fd.write(chunk)

        # unpack the files
        print('unpacking')
        zip_files = sorted(self.path.glob('*.zip'))
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.path)
            zip_file.unlink()

    def __load_metadata(self, random_state, rumble_only):
        metadata = pd.DataFrame()
        for stem, prefetch in self.prefetch.items():
            annotation_path = prefetch['annotation_path']
            annotations = pd.read_csv(annotation_path, sep='\t')
            if annotations.empty:
                continue

            # extract elephant rumbles
            file_metadata = Arden.__filter_annotations(annotations, stem)

            # extract background noise if applicable
            if not rumble_only:
                noise_metadata = Arden.__extract_noise(annotations, stem)
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
        for stem, prefetch in tqdm(self.prefetch.items()):
            # load in audio file
            audio_file = prefetch['audio_path']
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
        df['duration'] = df['End Time (s)'] - df['Begin Time (s)']
        return df
        
    @staticmethod
    def __extract_noise(rumble_annotations, stem):
        # get ranges for all possible rumbles
        rumble_event_ranges = list(zip(
            rumble_annotations['Begin Time (s)'],
            rumble_annotations['End Time (s)']
        ))

        # get average duration of rumbles
        rumble_deltas = [end - start for start, end in rumble_event_ranges]
        rumble_delta_stats = np.mean(rumble_deltas), np.std(rumble_deltas)

        # generate noise clips from regions without rumbles
        noise_regions = Arden.__find_noise_regions(rumble_event_ranges)
        clip_ranges = [
            Arden.__random_clip_range(region, rumble_delta_stats) 
            for region in noise_regions
        ]

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
                'earflap': 0,
                'duration': clip_end - clip_start
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
            & (df['duration'] > 2)
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
