import logging
from pathlib import Path
from typing import Union, List, Tuple

import cv2 as cv
import numpy as np

from ..feature_utils import load_features, unpack_features, save_features, pack_features
from ..types import PathType, FeaturesType, PackType


class FeatureExtractor:

    method = 'default'

    def __init__(self, enable_cache: bool = True):
        self.enable_cache = enable_cache
        self.logger = logging.getLogger(__name__)

    def get_image_features(self,
                           image_path: PathType,
                           return_pack: bool = False
                           ) -> Union[FeaturesType, PackType]:
        pass

    def get_cache_features(self,
                           cache_path: PathType,
                           return_pack: bool = False,
                           ) -> Union[FeaturesType, PackType]:
        features = load_features(str(cache_path))
        return features if return_pack else unpack_features(features)

    def get_features(self,
                     image_path: PathType,
                     cache_path: PathType = None,
                     return_pack: bool = False,
                     ) -> Union[FeaturesType, PackType, None]:
        if self.enable_cache and cache_path and Path(cache_path).exists():
            try:
                return self.get_cache_features(str(cache_path), return_pack=return_pack)
            except ValueError:
                self.logger.warning(f'读取缓存({str(cache_path)})失败，尝试进行计算')
        if Path(image_path).exists():
            features = self.get_image_features(str(image_path), return_pack=return_pack)
            if cache_path is not None:
                save_features(cache_path, features if return_pack else pack_features(*features))
            return features
        else:
            self.logger.warning(f'目标文件({str(image_path)}不存在，无法计算)')
            return None


class FeatureMatcher:

    def get_match_contours(self,
                           src_shape: Tuple[int, int, int],
                           src_features: FeaturesType,
                           dst_features: FeaturesType,
                           ) -> List[np.ndarray]:
        pass


class Matches:

    def __init__(self, matches: Union[List[cv.DMatch], np.recarray, np.ndarray]):
        self._matches = matches
        self._dst_contours = []

    @property
    def matches(self) -> Union[List[cv.DMatch], np.recarray, np.ndarray]:
        return self._matches

    @property
    def dst_contours(self) -> List[np.ndarray]:
        return self._dst_contours

    def update(self,
               src_pts: np.ndarray,
               dst_pts: np.ndarray,
               src_cnt: np.ndarray,
               ) -> bool:
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if M is None:
            return False
        dst = cv.perspectiveTransform(src_cnt, M)
        s = cv.matchShapes(src_cnt, dst, cv.CONTOURS_MATCH_I1, 0.000)
        if s < 0.05:
            self._dst_contours.append(np.int32(dst))
        self._matches = np.array(self._matches)[np.logical_not(mask.ravel().tolist())]
        return True
