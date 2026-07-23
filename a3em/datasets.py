import os
from abc import ABC
from pathlib import Path

class Arden:

    _doi: str = 'doi%253A10.5061%252Fdryad.7rh4625'
    
    # TODO
    @classmethod
    def load_data(
            api_token: str, path: str, test_split: float, 
            random_state: int, shuffle: bool) -> tuple:
        return None,

    # FIXME
    @classmethod
    def __prefetch(path: str) -> dict:
        path = Path(path)
        annotations_path = path.joinpath('manutalAnnotations')
        audio_path = path.joinpath('clips')
        if __validate_local_data(path):
            res = {}
            return res
            
        # make sure the directory is clear
        a3em.utils.clean_directory(path)

        # obtain token

        # download data
        res = requests.get(
            f'https://datadryad.org/api/v2/datasets{_doi}/download')
        
        return {}

    # TODO
    @classmethod
    def __validate_local_data(path: str) -> bool:
        return False

    # TODO
    @classmethod
    def iter() -> tuple:
        return None,

class Dataset(ABC):

    @classmethod
    @abstractmethod
    def load_data(
            path: str, test_split: float, 
            random_state: int, shuffle: bool) -> tuple:
        return None,

    @classmethod
    @abstractmethod
    def __prefetch(path: str) -> dict:
        return {}

    @classmethod
    @abstractmethod
    def __validate_local_data(path: str) -> bool:
        return False

    @classmethod
    @abstractmethod
    def iter() -> tuple:
        return None,

Dataset.register(Arden)
