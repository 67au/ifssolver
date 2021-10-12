import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple

PathType = Union[Path, str]
FeatureType = Tuple[List[cv.KeyPoint], np.ndarray]
MatchType = List[Tuple[cv.DMatch, cv.DMatch]]
