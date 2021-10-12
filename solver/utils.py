import hashlib
from typing import List, Tuple

import cv2 as cv
import numpy as np

from .types import PathType


def parse_portal_filename(lat: float, lng: float, image_url: str):
    return f'{lat}_{lng}_{hashlib.md5(image_url.encode()).hexdigest()}.jpg'


def get_picture_max_border(image_path: PathType, thresh=200) -> Tuple[int, int]:
    img = cv.imread(str(image_path))
    std = np.array([50, 50, 50])
    img_binary = cv.inRange(np.sum(np.power(img - std, 2), axis=2), 0, thresh)
    contours: List[np.ndarray]
    contours, _ = cv.findContours(255 - img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pt_max = [np.max(cnt, axis=0) for cnt in contours]
    xy_max = np.max(np.array(pt_max), axis=0)
    x, y = xy_max.ravel()
    return x, y
