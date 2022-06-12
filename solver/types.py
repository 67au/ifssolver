from pathlib import Path
from typing import Union, List, Tuple

import cv2 as cv
import numpy as np

PathType = Union[Path, str]
KeypointsType = List[Union[cv.KeyPoint, np.recarray]]
FeaturesType = Union[Tuple[KeypointsType, np.ndarray], np.recarray]
PackType = np.ndarray
