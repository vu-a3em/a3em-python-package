import a3em.utils
import os
import requests
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

class Arden:

    _api: str = 'https://datadryad.org/'
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

    # TODO
    @staticmethod
    def iter() -> tuple:
        return None,

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
