from typing import List, Tuple

import cv2 as cv
import numpy as np

from .types import PathType


def get_passcode(lnglat_list: List[Tuple[int, List[Tuple[float, float]]]],
                 output_filename: PathType,
                 char_size: int = 80,
                 border: int = 10
                 ) -> None:
    canvas = np.zeros((char_size + border * 2, (char_size + border * 2) * len(lnglat_list), 1))
    for n, lnglat in lnglat_list:
        if not any(lnglat):
            continue
        xy = np.array(lnglat, dtype=np.float32)
        xy_norm = xy - np.min(xy, axis=0)
        xy_square = np.around((char_size / np.max(xy_norm)) * xy_norm)
        xy_canvas = np.zeros_like(xy_square)
        xy_canvas[:, 0] = xy_square[:, 0] + ((2 * n) + 1) * border + n * char_size
        xy_canvas[:, 1] = char_size - xy_square[:, 1] + border
        cv.polylines(canvas, np.int32([xy_canvas]), False, 255, 1, cv.LINE_AA)
    cv.imwrite(str(output_filename), canvas)


def get_picture_max_border(image_path: PathType, thresh: int = 200,
                           BGR_std: Tuple[int, int, int] = (50, 50, 50)
                           ) -> Tuple[int, int]:
    img = cv.imread(str(image_path))
    std = np.array(BGR_std)
    img_binary = cv.inRange(np.sum(np.power(img - std, 2), axis=2), 0, thresh)
    contours: list[np.ndarray]
    contours, _ = cv.findContours(255 - img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pt_max = [np.max(cnt, axis=0) for cnt in contours]
    xy_max = np.max(np.array(pt_max), axis=0)
    x, y = xy_max.ravel()
    return x, y


def get_cnt_center(contour: np.ndarray) -> Tuple[int, int]:
    MM = cv.moments(contour)
    return int(MM['m10'] / MM['m00']), int(MM['m01'] / MM['m00'])
