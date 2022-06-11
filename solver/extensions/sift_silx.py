from typing import Union

import cv2 as cv
import numpy as np
from silx.image import sift

from ..types import PathType, FeaturesType, PackType
from .base import Matches, FeatureExtractor, FeatureMatcher


class SiftExtractor(FeatureExtractor):

    method = 'silx'

    def __init__(self,
                 devicetype: str = 'all',
                 platformid: int = None,
                 deviceid: int = None,
                 enable_cache: bool = True,
                 ):
        super().__init__(enable_cache)
        self.devicetype = devicetype
        self.platformid = platformid
        self.deviceid = deviceid

    def get_image_features(self,
                           image_path: PathType,
                           **kwargs
                           ) -> FeaturesType:
        image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        siftp = sift.SiftPlan(
            template=image,
            init_sigma=1,
            devicetype=self.devicetype,
            platformid=self.platformid,
            deviceid=self.deviceid
        )
        return siftp.keypoints(image)

    def get_cache_features(self,
                           cache_path: PathType,
                           **kwargs
                           ) -> FeaturesType:
        return super().get_cache_features(cache_path, return_pack=True)

    def get_features(self,
                     image_path: PathType,
                     cache_path: PathType = None,
                     **kwargs
                     ) -> Union[FeaturesType, PackType]:
        return super().get_features(image_path, cache_path, return_pack=True)


class SiftMatcher(FeatureMatcher):

    def __init__(self,
                 devicetype: str = 'all',
                 platformid: int = None,
                 deviceid: int = None
                 ):
        self._matcher = sift.MatchPlan(
            size=65536,
            devicetype=devicetype,
            device=(platformid, deviceid),
        )

    def get_match_contours(self,
                           src_shape: tuple[int, int, int],
                           src_features: FeaturesType,
                           dst_features: FeaturesType,
                           ) -> list[np.ndarray]:
        h, w, _ = src_shape
        src_cnt = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        matches = Matches(self._matcher.match(src_features, dst_features))
        while len(matches.matches) >= 4:
            src_des, dst_des = matches.matches[:, 0], matches.matches[:, 1]
            src_pts = src_des[['x', 'y']].astype([('x', '<f4'), ('y', '<f4')]).view('<f4').reshape(-1, 2)
            dst_pts = dst_des[['x', 'y']].astype([('x', '<f4'), ('y', '<f4')]).view('<f4').reshape(-1, 2)
            if not matches.update(src_pts, dst_pts, src_cnt):
                break
        return matches.dst_contours
