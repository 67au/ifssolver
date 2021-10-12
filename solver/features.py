from typing import List, Tuple, Union

import cv2 as cv
import numpy as np

from .types import PathType


class FeatureUtils:

    @classmethod
    def pack_features(cls,
                      keypoints: List['cv.KeyPoint'],
                      descriptors: 'np.ndarray'
                      ) -> 'np.ndarray':
        kp = np.array([
            (kp.pt[0], kp.pt[1], kp.angle, kp.class_id, kp.octave, kp.response, kp.size) for kp in keypoints
        ])
        des = np.array(descriptors)
        return np.hstack((kp, des))

    @classmethod
    def unpack_features(cls, array: 'np.ndarray') -> Tuple[List['cv.KeyPoint'], 'np.ndarray']:
        kp = array[:, :7]
        des = array[:, 7:]
        keypoints = [
            cv.KeyPoint(
                x=x, y=y, angle=angle, class_id=int(class_id), octave=int(octave), response=response, size=size
            ) for x, y, angle, class_id, octave, response, size in kp
        ]
        descriptors = np.array(des).astype(np.float32)
        return keypoints, descriptors

    @classmethod
    def save_features(cls, filename: PathType, array: Union['np.ndarray', 'np.recarray']):
        np.save(str(filename), array)

    @classmethod
    def load_features(cls, filename: PathType) -> Union['np.ndarray', 'np.recarray']:
        return np.load(str(filename))
