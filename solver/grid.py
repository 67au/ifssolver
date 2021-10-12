from collections import defaultdict
from typing import List, Union

import cv2 as cv
import numpy as np


class GridUtils:

    @classmethod
    def grid_sort(cls,
                  x_list: List[Union[float, int]],
                  y_list: List[Union[float, int]],
                  column_num: int,
                  ):
        x_array = np.array(x_list).astype(np.float32)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, _ = cv.kmeans(x_array, column_num, None, criteria, 10, cv.KMEANS_PP_CENTERS)
        grid = defaultdict(list)
        for n, m in enumerate(label.ravel()):
            grid[m].append(n)
        return [sorted(v, key=lambda k: y_list[k]) for v in sorted(grid.values(), key=lambda k: x_list[k[0]])]

