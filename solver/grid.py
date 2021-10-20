from collections import defaultdict
from typing import List, Union

import cv2 as cv
import numpy as np


class GridUtils:

    @classmethod
    def grid_sort(cls,
                  xy_array: np.ndarray,
                  column_num: int,
                  ):
        x_array = xy_array[:, 0].astype(np.float32)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, _ = cv.kmeans(x_array, column_num, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        grid = defaultdict(list)
        for n, m in enumerate(label.ravel()):
            grid[m].append(n)
        return [sorted(v, key=lambda k: xy_array[:, 0].astype(np.float32).ravel()[k])
                for v in sorted(grid.values(), key=lambda k: x_array.ravel()[k[0]])]

