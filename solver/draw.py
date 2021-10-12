from typing import List, Tuple

import cv2 as cv
import numpy as np

from .types import PathType


class DrawUtils:

    @classmethod
    def get_passcode(cls,
                     lnglat_list: List[Tuple[int, List[Tuple[float, float]]]],
                     output_filename: PathType,
                     char_size: int = 80,
                     border: int = 10
                     ):
        canvas = np.zeros((char_size + border * 2, (char_size + border * 2) * len(lnglat_list), 1))
        for n, lnglat in lnglat_list:
            xy = np.array(lnglat, dtype=np.float32)
            xy_norm = xy - np.min(xy, axis=0)
            xy_square = np.around((char_size / np.max(xy_norm)) * xy_norm)
            xy_canvas = np.zeros_like(xy_square)
            xy_canvas[:, 0] = xy_square[:, 0] + ((2 * n) + 1) * border + n * char_size
            xy_canvas[:, 1] = char_size - xy_square[:, 1] + border
            cv.polylines(canvas, np.int32([xy_canvas]), False, 255, 1, cv.LINE_AA)
        cv.imwrite(str(output_filename), canvas)