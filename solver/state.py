import hashlib
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .types import PathType


class MatchState:

    def __init__(self, state_path: PathType, metadata_path: PathType, save_progress: bool = True):
        self.state_path = state_path
        self.save_progress = save_progress
        metadata_digest = self.get_file_hash(metadata_path)
        if not self.save_progress:
            Path(self.state_path).unlink(missing_ok=True)
        if Path(self.state_path).exists():
            with open(self.state_path, 'rb') as f:
                self._state = pickle.load(f)
            if self.metadata_digest == metadata_digest:
                return
        self._state = {
            'metadata_digest': '',
            'index': 0,
            'match_cnts': [],
        }
        self.metadata_digest = metadata_digest

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.save_progress:
            with open(self.state_path, 'wb') as f:
                pickle.dump(self._state, f)

    @property
    def metadata_digest(self) -> str:
        return self._state.get('metadata_digest', '')

    @metadata_digest.setter
    def metadata_digest(self, value):
        self._state['metadata_digest'] = value

    @property
    def index(self) -> int:
        return self._state.get('index', 0)

    @index.setter
    def index(self, value):
        self._state['index'] = value

    @property
    def match_cnts(self) -> List[Tuple[int, np.ndarray]]:
        return self._state.get('match_cnts', [])

    @staticmethod
    def get_file_hash(file_path: PathType):
        with open(file_path, 'rb') as f:
            digest = hashlib.file_digest(f, 'sha256')
        return digest.hexdigest()

    def save_index(self, index: int):
        self.index = index

    def save_cnt(self, num: int, cnt: np.ndarray):
        self.match_cnts.append((num, cnt))
