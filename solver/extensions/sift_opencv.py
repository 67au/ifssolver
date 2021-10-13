from pathlib import Path
from typing import Union, List

import cv2 as cv
import numpy as np

from ..features import FeatureUtils
from ..types import PathType, FeatureType, MatchType


class SIFTUtils:
    _sift = cv.SIFT_create()

    @classmethod
    def get_image_features(cls,
                           image_path: PathType,
                           ) -> FeatureType:
        image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        kp, des = cls._sift.detectAndCompute(image, None)
        return kp, des

    @classmethod
    def get_cache_features(cls,
                           cache_path: PathType,
                           pack: bool = False,
                           ) -> Union[FeatureType, 'np.ndarray']:
        features = FeatureUtils.load_features(cache_path)
        if pack:
            return features
        kp, des = FeatureUtils.unpack_features(features)
        return kp, des

    @classmethod
    def get_features(cls,
                     image_path: PathType,
                     feature_dir: PathType,
                     save_cache: bool = True,
                     no_clean: bool = False,
                     pack: bool = False,
                     ) -> Union[FeatureType, 'np.ndarray']:
        image_path = Path(image_path)
        feature_dir = Path(feature_dir)
        if no_clean:
            cache_path = feature_dir.joinpath(f'{image_path.name}.npy')
            if cache_path.exists():
                try:
                    if pack:
                        features = cls.get_cache_features(cache_path, pack)
                        return features
                    kp, des = cls.get_cache_features(cache_path)
                    return kp, des
                except ValueError:
                    cache_path.unlink(missing_ok=True)
        if image_path.exists():
            kp, des = cls.get_image_features(image_path)
            feature_dir.mkdir(exist_ok=True, parents=True)
            cache_path = feature_dir.joinpath(f'{image_path.name}.npy')
            features = None
            if save_cache and not (cache_path.exists() and no_clean):
                features = FeatureUtils.pack_features(kp, des)
                FeatureUtils.save_features(cache_path, features)
            if pack:
                return FeatureUtils.pack_features(kp, des) if features is None else features
            return kp, des
        raise


class BFMatcher:

    def __init__(self):
        self._matcher = cv.BFMatcher_create()

    def get_matches(self, src_des: 'np.ndarray', dst_des: 'np.ndarray') -> List[cv.DMatch]:
        matches: MatchType = self._matcher.knnMatch(src_des, dst_des, k=2)
        best_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]
        return best_matches

    def get_centers(self,
                    src_image: np.ndarray,
                    src_features: Union[FeatureType, np.ndarray],
                    dst_image: np.ndarray,
                    dst_features: Union[FeatureType, np.ndarray]
                    ):
        src_kp, src_des = src_features if isinstance(src_features, tuple) else \
            FeatureUtils.unpack_features(src_features)
        dst_kp, dst_des = dst_features if isinstance(dst_features, tuple) else \
            FeatureUtils.unpack_features(dst_features)
        matches = self.get_matches(src_des, dst_des)
        centers = []
        for _ in iter(lambda: len(matches) < 4, True):
            src_pts = np.float32([src_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 35.0)
            h, w, _ = src_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            if M is None:
                return centers
            dst: np.ndarray = cv.perspectiveTransform(pts, M)
            s = cv.matchShapes(pts, dst, cv.CONTOURS_MATCH_I1, 0.000)
            if s < 0.05:
                MM = cv.moments(dst)
                cx = int(MM['m10'] / MM['m00'])
                cy = int(MM['m01'] / MM['m00'])
                centers.append((cx, cy))
                dst_image = cv.polylines(dst_image, [np.int32(dst)], True, (0, 0, 255), 1, cv.LINE_AA)
            matches = np.array(matches)[np.logical_not(mask.ravel().tolist())].tolist()
        return centers
