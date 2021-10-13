from pathlib import Path
from typing import Optional, List, Tuple

import cv2 as cv
import numpy as np
from silx.image import sift

from ..features import FeatureUtils
from ..types import PathType


class SIFTUtils:

    def __init__(self,
                 devicetype: str = 'all',
                 platformid: int = None,
                 deviceid: int = None
                 ):
        self.devicetype = devicetype
        self.platformid = platformid
        self.deviceid = deviceid

    def get_image_features(self,
                           image_path: PathType,
                           ) -> np.recarray:
        image = cv.imread(str(image_path))
        siftp = sift.SiftPlan(
            template=image,
            init_sigma=1.2,
            devicetype=self.devicetype,
            platformid=self.platformid,
            deviceid=self.deviceid,
        )
        kp = siftp.keypoints(image)
        return kp

    def get_cache_features(self,
                           cache_path: PathType,
                           ) -> np.recarray:
        kp = FeatureUtils.load_features(cache_path)
        return kp

    def get_features(self,
                     image_path: PathType,
                     feature_dir: PathType,
                     save_cache: bool = True,
                     no_clean: bool = False,
                     ) -> np.recarray:
        image_path = Path(image_path)
        feature_dir = Path(feature_dir)
        if no_clean:
            cache_path = feature_dir.joinpath(f'{image_path.name}.npy')
            if cache_path.exists():
                try:
                    kp = self.get_cache_features(cache_path)
                    return kp
                except ValueError:
                    cache_path.unlink(missing_ok=True)
        if image_path.exists():
            kp = self.get_image_features(image_path)
            feature_dir.mkdir(exist_ok=True, parents=True)
            cache_path = feature_dir.joinpath(f'{image_path.name}.npy')
            if save_cache and not (cache_path.exists() and no_clean):
                FeatureUtils.save_features(cache_path, kp)
            return kp
        raise


class SIFTMatcher:

    def __init__(self,
                 devicetype: str = 'all',
                 platformid: int = None,
                 deviceid: int = None
                 ):
        self._sift = sift.MatchPlan(
            devicetype=devicetype,
            device=(platformid, deviceid)
        )

    def get_centers(self,
                    src_image: np.ndarray,
                    src_keypoints: np.recarray,
                    dst_image: np.ndarray,
                    dst_keypoints: np.recarray
                    ) -> Optional[List[Tuple[float, float]]]:
        centers = []
        matches = self._sift.match(src_keypoints, dst_keypoints)
        for _ in iter(lambda: len(matches) < 4, True):
            src_des, dst_des = matches[:, 0], matches[:, 1]
            src_pts = src_des[['x', 'y']].astype([('x', '<f8'), ('y', '<f8')]).view('<f8').reshape(-1, 2)
            dst_pts = dst_des[['x', 'y']].astype([('x', '<f8'), ('y', '<f8')]).view('<f8').reshape(-1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 35.0)
            h, w, _ = src_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            if M is None:
                return centers
            dst: np.ndarray = cv.perspectiveTransform(pts, M)
            s = cv.matchShapes(pts, dst, cv.CONTOURS_MATCH_I1, 0.000)
            if s < 0.03:
                MM = cv.moments(dst)
                cx = int(MM['m10'] / MM['m00'])
                cy = int(MM['m01'] / MM['m00'])
                centers.append((cx, cy))
                dst_image = cv.polylines(dst_image, [np.int32(dst)], True, (0, 0, 255), 1, cv.LINE_AA)
            matches = matches[np.logical_not(mask.ravel().tolist())]
        return centers
