from collections import defaultdict

import cv2 as cv
import numpy as np


def sort_grid(xy_array: np.ndarray, column_num: int) -> list:
    column_num = min(xy_array.shape[0], column_num)
    x_array = xy_array[:, 0].astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, _ = cv.kmeans(x_array, column_num, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    grid = defaultdict(list)
    for n, m in enumerate(label.ravel()):
        grid[m].append(n)
    return [sorted(v, key=lambda k: xy_array[:, 1].astype(np.float32).ravel()[k])
            for v in sorted(grid.values(), key=lambda k: x_array.ravel()[k[0]])]
