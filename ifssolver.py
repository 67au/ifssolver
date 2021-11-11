#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import asyncio
import csv
import logging.config
from configparser import ConfigParser
from itertools import groupby
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np
import tqdm
import tqdm.contrib.concurrent

from solver import PortalUtils, GridUtils, DrawUtils
from solver.config import ConfigProxy
from solver.types import PathType
from solver.utils import parse_portal_filename, get_picture_max_border

logger = logging.getLogger('main')


async def download_csv(config: ConfigProxy):
    async with PortalUtils.get_intel_client(config.cookies, config.proxy) as client:
        portals = await PortalUtils.get_portals_from_square(client, config.lat, config.lng, config.radius)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, partial(PortalUtils.save_portals_as_csv, config.metadata_csv, portals))


async def download_images(config: ConfigProxy, no_clean: bool):
    if config.proxy is not None:
        logger.info('如果使用代理，将会限制连接数为1，详情查看 https://github.com/encode/httpcore/issues/335')
    await PortalUtils.download_portals_by_csv(config.metadata_csv, config.portal_images_dir,
                                              proxy=config.proxy, no_clean=no_clean, max_worker=20)


def save_match_result(filename: PathType, result_list: Iterable):
    with open(filename, 'w', newline='', encoding='utf8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['col', 'row', 'x', 'y', 'title', 'lat', 'lng', 'image'])
        f_csv.writerows(result_list)


def read_match_result(filename: PathType):
    with open(filename, 'r', encoding='utf8') as f:
        f_csv = csv.DictReader(f)
        return list(f_csv)


def split_picture(config: ConfigProxy, no_clean: bool = False, save_cache: bool = True, method: str = 'opencv'):
    if method == 'silx':
        from solver.extensions.sift_silx import SIFTUtils, SIFTMatcher
        silx_config = config.silx
        compute_utils = SIFTUtils(**silx_config)
        matcher = SIFTMatcher(**silx_config)
    elif method == 'opencv':
        logger.info('有显卡的用户推荐使用 --method silx')
        from solver.extensions.sift_opencv import SIFTUtils, BFMatcher
        compute_utils = SIFTUtils()
        matcher = BFMatcher()
    else:
        raise

    logger.info('计算 IFS 图像')
    x, y = get_picture_max_border(config.ifs_image_path)
    ifs_image = cv.imread(str(config.ifs_image_path))
    ifs_image_crop = ifs_image[0:y, 0:x]
    ifs_image_crop_path = config.output_sub_dir\
        .joinpath(f'{str(config.ifs_image_path.stem)}_{x}_{y}{config.ifs_image_path.suffix}')
    cv.imwrite(str(ifs_image_crop_path), ifs_image_crop)
    ifs_image_features = compute_utils.get_features(ifs_image_crop_path, feature_dir=config.portal_features_dir,
                                                    no_clean=no_clean, save_cache=save_cache)

    logger.info('计算 Portal 图像')
    portals = PortalUtils.read_portals_from_csv(config.metadata_csv)
    result_list = []
    for n, p in enumerate(tqdm.tqdm(portals)):
        portal_image_path = config.portal_images_dir.joinpath(parse_portal_filename(
            lat=p['lat'], lng=p['lng'], image_url=p['image']
        ))
        portal_image = cv.imread(str(portal_image_path))
        portal_features = compute_utils.get_features(
            portal_image_path,
            feature_dir=config.portal_features_dir,
            no_clean=no_clean,
            save_cache=save_cache,
        )
        centers = matcher.get_centers(portal_image, portal_features, ifs_image, ifs_image_features)
        if any(centers):
            for lat, lng in centers:
                result_list.append((lat, lng, n))
    xy_array = np.array(result_list)[:, 0:2]
    grid_list = GridUtils.grid_sort(xy_array, config.column)
    result_iter = ((k, n, result_list[val][0], result_list[val][1], portals[result_list[val][2]]['title'],
                    portals[result_list[val][2]]['lat'], portals[result_list[val][2]]['lng'],
                    portals[result_list[val][2]]['image'])
                   for k, v in enumerate(grid_list) for n, val in enumerate(v))
    save_match_result(config.match_result_csv, result_iter)
    cv.imwrite(str(config.match_result_jpg), ifs_image)


def draw_result(config: ConfigProxy):
    output_sub_dir = config.output_sub_dir
    if not config.match_result_csv.exists():
        logger.error(f'匹配结果 {str(config.match_result_csv)} 不存在，请先使用 --split 参数')
        return
    result_list = read_match_result(config.match_result_csv)
    result_list.sort(key=itemgetter('col'))
    cols = groupby(result_list, key=itemgetter('col'))
    lnglat_list = [(int(k), [(d['lng'], d['lat']) for d in sorted(v, key=itemgetter('row'))]) for k, v in cols]
    DrawUtils.get_passcode(lnglat_list, str(config.passcode_jpg))
    print(f'结果已输出到 {str(output_sub_dir)}，结合匹配情况对比目标图像判断，以修正记录：')
    print('\n'.join((f' 第 {n+1:2} 列匹配个数: {len(l):2}' for n, l in sorted(lnglat_list, key=lambda k: k[0]))))


def main():
    parser = argparse.ArgumentParser(description='ifssolver')
    parser.add_argument('--config', dest='config', metavar='filename', default='config.ini',
                        action='store', help='Configure file, default = \'config.ini\'', required=False)

    group1 = parser.add_mutually_exclusive_group()

    group11 = group1.add_mutually_exclusive_group()
    group11.add_argument('--download-csv', help='download metadata csv only', action='store_true')
    group11.add_argument('--download-img', help='download image for metadata csv', action='store_true')
    group11.add_argument('--download-all', help='download image after updating metadata csv', action='store_true')

    group1.add_argument('--split', help='split ifs picture only', action='store_true')

    group1.add_argument('--draw', help='draw result only', action='store_true')

    group1.add_argument('--auto', help='AUTO', action='store_true')

    parser.add_argument('--method', dest='method', metavar='opencv', default='opencv',
                        action='store', help='sift algorithm provider, opencv or silx', required=False)

    parser.add_argument('--no-clean', help='use the last files', action='store_true')

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f'无法找到配置文件 {str(config_path)}')
        return
    config = ConfigProxy.load_config(config_path)

    if args.auto:
        logger.info(f'使用配置文件 {args.config} 进行自动处理')

    if args.download_csv or args.download_all or args.auto:
        logger.info('下载指定区域的 Portal 元数据')
        asyncio.run(download_csv(config))
    if args.download_img or args.download_all or args.auto:
        logger.info('下载指定区域的 Portal 照片')
        asyncio.run(download_images(config, args.no_clean))

    if args.split or args.auto:
        logger.info('识别图中的 Portal 照片')
        split_picture(config, no_clean=args.no_clean, method=args.method)

    if args.draw or args.auto:
        logger.info('生成 Passcode 图像')
        draw_result(config)


if __name__ == '__main__':
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {
                'format': '%(asctime)s %(levelname)s %(message)s',
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'default',
            },
        },
        'loggers': {
            'main': {
                'handlers': ['console'],
                'level': 'INFO',
            },
        }
    })
    main()
