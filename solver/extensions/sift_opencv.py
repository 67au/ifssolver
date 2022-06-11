from typing import Union, Tuple, List

import cv2 as cv
import numpy as np

from ..feature_utils import pack_features
from ..types import PathType, FeaturesType, PackType
from .base import Matches, FeatureExtractor, FeatureMatcher


class SiftExtractor(FeatureExtractor):

    method = 'opencv'

    def __init__(self, enable_cache: bool = True):
        super().__init__(enable_cache)
        self._sift = cv.SIFT_create()

    def get_image_features(self,
                           image_path: PathType,
                           return_pack: bool = False
                           ) -> Union[FeaturesType, PackType]:
        image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        kp, des = self._sift.detectAndCompute(image, None)
        return pack_features(kp, des) if return_pack else (kp, des)

    def get_cache_features(self,
                           cache_path: PathType,
                           return_pack: bool = False,
                           ) -> Union[FeaturesType, PackType]:
        return super().get_cache_features(cache_path, return_pack=return_pack)

    def get_features(self,
                     image_path: PathType,
                     cache_path: PathType = None,
                     return_pack: bool = False,
                     ) -> Union[FeaturesType, PackType, None]:
        return super().get_features(image_path, cache_path, return_pack=return_pack)


class BFMatcher(FeatureMatcher):

    def __init__(self):
        self._matcher = cv.BFMatcher_create()

    def get_match_contours(self,
                           src_shape: Tuple[int, int, int],
                           src_features: FeaturesType,
                           dst_features: FeaturesType,
                           ) -> List[np.ndarray]:
        src_kp, src_des = src_features
        dst_kp, dst_des = dst_features
        h, w, _ = src_shape
        src_cnt = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        matches = Matches(
            np.array(sorted(
                (m for m, n in self._matcher.knnMatch(src_des, dst_des, k=2) if m.distance < 0.75 * n.distance),
                key=lambda m: m.distance
            ))
        )
        while len(matches.matches) >= 4:
            src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches.matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches.matches]).reshape(-1, 1, 2)
            if not matches.update(src_pts, dst_pts, src_cnt):
                break
        return matches.dst_contours
