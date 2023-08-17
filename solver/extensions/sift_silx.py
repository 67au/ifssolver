from functools import lru_cache
from typing import Union, Tuple, List

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

    @lru_cache(maxsize=128)
    def _create_sift_plan(self, shape, dtype) -> sift.SiftPlan:
        return sift.SiftPlan(
            shape=shape,
            dtype=dtype,
            devicetype=self.devicetype,
            platformid=self.platformid,
            deviceid=self.deviceid
        )

    def get_image_features(self,
                           image_path: PathType,
                           return_pack: bool = False,
                           ) -> FeaturesType:
        image = self.get_image(image_path)
        if max(image.shape) > 768:
            siftp = self._create_sift_plan.__wrapped__(self, image.shape, image.dtype)
        else:
            siftp = self._create_sift_plan(image.shape, image.dtype)
        return siftp.keypoints(image)

    def get_cache_features(self,
                           cache_path: PathType,
                           **kwargs
                           ) -> FeaturesType:
        return super().get_cache_features(cache_path, return_pack=True)

    def get_features(self,
                     image_path: PathType,
                     cache_path: PathType = None,
                     return_pack: bool = True,
                     ) -> Union[FeaturesType, PackType]:
        return super().get_features(image_path, cache_path, return_pack=True)

    def get_features_and_shape(self,
                               image_path: PathType,
                               cache_path: PathType = None,
                               return_pack: bool = True,
                               scale: bool = True,
                               ) -> Tuple[Union[FeaturesType, PackType, None], tuple]:
        return super().get_features_and_shape(image_path, cache_path, return_pack=True)


class SiftMatcher(FeatureMatcher):

    def __init__(self,
                 devicetype: str = 'all',
                 platformid: int = None,
                 deviceid: int = None
                 ):
        self._matcher = sift.MatchPlan(
            devicetype=devicetype,
            device=(platformid, deviceid),
        )

    def get_match_contours(self,
                           src_shape: Tuple[int, int, int],
                           src_features: FeaturesType,
                           dst_features: FeaturesType,
                           ) -> List[np.ndarray]:
        h, w, *_ = src_shape
        src_cnt = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst_contours = []
        kp = dst_features
        while True:
            matches = self._matcher.match(src_features, kp)
            if len(matches) < 4:
                break
            src_des, dst_des = matches[:, 0], matches[:, 1]
            src_pts = src_des[['x', 'y']].astype([('x', '<f4'), ('y', '<f4')]).view('<f4').reshape(-1, 2)
            dst_pts = dst_des[['x', 'y']].astype([('x', '<f4'), ('y', '<f4')]).view('<f4').reshape(-1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            if M is None:
                break
            dst = cv.perspectiveTransform(src_cnt, M)
            s = cv.matchShapes(src_cnt, dst, cv.CONTOURS_MATCH_I1, 0.000)
            if s < 0.05:
                dst_contours.append(np.int32(dst))
            x_max, x_min = dst[:, :, 0].max(), dst[:, :, 0].min()
            y_max, y_min = dst[:, :, 1].max(), dst[:, :, 1].min()
            kp = kp[np.where(np.logical_not(
                np.logical_and.reduce((kp.x < x_max, kp.x > x_min, kp.y < y_max, kp.y > y_min))))]

        return dst_contours
