import asyncio
import csv
import logging
import sys
from functools import partial
from itertools import groupby
from operator import itemgetter
from typing import Callable, Tuple, List, Iterator

import aiofiles
import cv2 as cv
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .config import ConfigProxy
from .draw_utils import get_picture_max_border, get_cnt_center, get_passcode
from .intel_map import PortalDownloader
from .grid_utils import sort_grid
from .types import PathType
from .utils import parse_cache_path, parse_portal_filename

from .extensions.base import FeatureExtractor

MAX_WORKERS = 8


async def run_in_executor(func):
    if sys.version_info >= (3, 9):
        return await asyncio.to_thread(func)
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func)


class Solver:

    def __init__(self,
                 config: ConfigProxy,
                 no_clean: bool = False,
                 metadata: PathType = None,
                 ):
        self.config = config
        self.no_clean = no_clean
        self.metadata = metadata or config.metadata_csv
        self._downloader = PortalDownloader(
            image_dir=config.portal_images_dir,
            proxy_url=config.proxy,
            no_clean=no_clean,
        )

        self.logger = logging.getLogger(__name__)

    async def download_csv(self):
        portals = await self._downloader.iter_portals_by_square(
            self.config.cookies,
            self.config.lat,
            self.config.lng,
            self.config.radius
        )
        await run_in_executor(partial(self._downloader.save_portals_as_csv, self.config.metadata_csv, portals))

    async def download_images(self):
        portals_list = await run_in_executor(partial(self._downloader.read_portals_from_csv, self.metadata))
        ok, err = await self._downloader.download_portals_by_list(portals_list)
        if not ok:
            self.logger.warning(f'有{len(err)}个图像下载失败，可以尝试使用 --no-clean 参数下载失败部分')
            async with aiofiles.open(self.config.download_errors_txt, 'w', encoding='utf-8') as f:
                await f.writelines(f"{n}, {portals_list[n]['Name']}, \"{e}\"\n" for n, e in err)
            self.logger.warning(f'下载错误已保存在 {str(self.config.download_errors_txt)}')

    def _get_ifs_image_features(self, extractor: FeatureExtractor):
        x, y = get_picture_max_border(self.config.ifs_image_path)
        ifs_image = cv.imread(str(self.config.ifs_image_path))
        ifs_image_crop = ifs_image[0:y, 0:x]
        ifs_image_crop_path = self.config.output_sub_dir \
            .joinpath(f'{str(self.config.ifs_image_path.stem)}_{x}_{y}{self.config.ifs_image_path.suffix}')
        cv.imwrite(str(ifs_image_crop_path), ifs_image_crop)
        return extractor.get_image_features(ifs_image_crop_path)

    def _get_portal_image_features(self, extractor: FeatureExtractor, portal_image_path: PathType):
        shape = cv.imread(str(portal_image_path)).shape
        features = extractor.get_features(
            portal_image_path,
            parse_cache_path(self.config.portal_features_dir, extractor.method, portal_image_path)
        )
        return features, shape

    def _get_match(self,
                   extractor: FeatureExtractor,
                   portal_image_path: PathType,
                   matcher_func: Callable
                   ) -> List[np.ndarray]:
        features, shape = self._get_portal_image_features(extractor, portal_image_path)
        match = matcher_func(src_features=features, src_shape=shape)
        return match

    def _get_matches(self,
                     portals: List[dict],
                     extractor: FeatureExtractor,
                     matcher_func: Callable,
                     ) -> Iterator[Tuple[int, np.ndarray]]:
        errors_list = []
        with logging_redirect_tqdm():
            for num, p in enumerate(tqdm(portals)):
                portal_image_path = self.config.portal_images_dir.joinpath(
                    parse_portal_filename(p['Image'], p['Latitude'], p['Longitude']))
                if not portal_image_path.exists():
                    self.logger.debug(f"Portal 照片不存在: ({num}) {p['Name']}")
                    errors_list.append((num, 'Not Found'))
                else:
                    s = self._get_match(extractor, portal_image_path, matcher_func)
                    for m in s:
                        yield num, m

        if any(errors_list):
            with open(self.config.split_errors_txt, 'w', encoding='utf-8') as f:
                f.writelines(f"{n}, {portals[n]['Name']}, \"{e}\"\n" for n, e in errors_list)
            self.logger.warning(f'有 {len(errors_list)} 张 Portal 照片无法计算，请查看 {str(self.config.split_errors_txt)}')

    MATCH_FIELD = ['col', 'row', 'lat', 'lng', 'x', 'y', 'name']

    def _save_match_result(self, result: Iterator[tuple]):
        with open(self.config.match_result_csv, 'w', newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(self.MATCH_FIELD)
            f_csv.writerows(result)

    def _read_match_result(self):
        with open(self.config.match_result_csv, 'r', newline='', encoding='utf-8') as f:
            f_csv = csv.DictReader(f)
            return list(f_csv)

    def _write_match_image(self, cnts: Iterator[np.ndarray]):
        img = cv.imread(str(self.config.ifs_image_path))
        for cnt in cnts:
            cv.polylines(img, [cnt], True, (0, 0, 255), 1, cv.LINE_AA)
        cv.imwrite(str(self.config.match_result_jpg), img)

    def _check_cache_dir(self, method: str):
        self.config.portal_features_dir.joinpath(method).mkdir(exist_ok=True)

    def split_picture(self, method: str):
        if method == 'silx':
            from solver.extensions.sift_silx import SiftExtractor, SiftMatcher
            extractor = SiftExtractor(**self.config.silx, enable_cache=self.no_clean)
            matcher = SiftMatcher(**self.config.silx)
        elif method == 'opencv':
            from solver.extensions.sift_opencv import SiftExtractor, BFMatcher
            extractor = SiftExtractor(enable_cache=self.no_clean)
            matcher = BFMatcher()
        else:
            self.logger.error(f'不支持使用 {method} 方法')
            sys.exit(0)

        self.logger.info('计算 IFS 图像')
        ifs_image_features = self._get_ifs_image_features(extractor)

        self._check_cache_dir(extractor.method)

        self.logger.info('计算 Portal 图像')
        portals = self._downloader.read_portals_from_csv(self.metadata)

        match_cnts = list(self._get_matches(
            portals,
            extractor,
            partial(matcher.get_match_contours, dst_features=ifs_image_features),
        ))

        centers = np.array([get_cnt_center(cnt[1]) for cnt in match_cnts])
        grids = sort_grid(centers, self.config.column)
        result = (
            (i, j, portals[match_cnts[v][0]]['Latitude'], portals[match_cnts[v][0]]['Longitude'],
             centers[v, 0], centers[v, 1], portals[match_cnts[v][0]]['Name'])
            for i, val in enumerate(grids, 1) for j, v in enumerate(val, 1)
        )

        self._save_match_result(result)
        self._write_match_image(cnt[1] for cnt in match_cnts)

    def draw_passcode(self):
        if not self.config.match_result_csv.exists():
            self.logger.error(f'匹配结果 {str(self.config.match_result_csv)} 不存在，请先使用 --split 识别')
            return
        result = self._read_match_result()
        result.sort(key=itemgetter('col'))
        cols = groupby(result, key=itemgetter('col'))
        centers = [(int(k) - 1, [(d['lng'], d['lat']) for d in sorted(v, key=itemgetter('row'))]) for k, v in cols]
        get_passcode(centers, str(self.config.passcode_jpg))
        self.logger.info(f'结果已输出到 {str(self.config.output_sub_dir)}，结合匹配情况对比目标图像判断')
