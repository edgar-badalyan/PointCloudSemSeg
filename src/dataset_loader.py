from abc import ABC, abstractmethod
from typing import List
import numpy as np


class DatasetLoader(ABC):

    @abstractmethod
    def __init__(self, dataset_dir: str):
        pass

    @abstractmethod
    def load_image(self, im_name: str, im_dir: str = None) -> np.ndarray:
        pass

    @abstractmethod
    def get_window_dicts(self, im_dir: str) -> List[dict]:
        pass

    @abstractmethod
    def parse_annotation(self, name: str) -> dict:
        pass
