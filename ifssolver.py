#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from solver import Solver
from solver.config import ConfigProxy

logger = logging.getLogger('ifssolver')


def main():
    parser = argparse.ArgumentParser(description='ifssolver')
    parser.add_argument('--config', dest='config', metavar='filename', default='config.ini',
                        action='store', help='Configure File, default = \'config.ini\'', required=False)

    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument('--download-csv', help='download metadata', action='store_true')
    download_group.add_argument('--download-img', help='download image by metadata', action='store_true')
    download_group.add_argument('--download-all', help='download image after updating metadata', action='store_true')

    split_group = parser.add_argument_group()
    split_group.add_argument('--split', help='split ifs image', action='store_true')
    split_group.add_argument('--draw', help='draw result', action='store_true')

    parser.add_argument('--metadata', dest='metadata', action='store', help='use specified METADATA')
    parser.add_argument('--method', dest='method', metavar='opencv', default='opencv',
                        action='store', help='sift algorithm provider, opencv or silx', required=False)
    parser.add_argument('--no-clean', help='no clean cache file', action='store_true')
    parser.add_argument('--save-progress', help='save split progress', action='store_true')

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f'无法找到配置文件({str(config_path)})')
        sys.exit(0)
    config = ConfigProxy.load_config(config_path)

    solver = Solver(config, args.no_clean, args.save_progress, metadata_csv=args.metadata)
    auto = False

    if not any((args.download_csv, args.download_img, args.download_all, args.split, args.draw)):
        logger.info(f'使用配置文件({args.config})进行自动处理')
        auto = True

    if args.download_csv or args.download_all or auto:
        logger.info('下载 Portal 元数据')
        asyncio.run(solver.download_csv())

    if args.download_img or args.download_all or auto:
        logger.info('下载 Portal 照片')
        asyncio.run(solver.download_images())

    if args.split or auto:
        logger.info('识别图中的 Portal 照片')
        asyncio.run(solver.split_picture(args.method))

    if args.draw or auto:
        logger.info('生成 Passcode 图像')
        solver.draw_passcode()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
    )
    logging.getLogger('silx.opencl.sift.plan').disabled = True
    logging.getLogger('silx.opencl.sift.match').disabled = True
    logging.getLogger('silx.opencl.processing').disabled = True
    try:
        main()
    except KeyboardInterrupt:
        exit()
