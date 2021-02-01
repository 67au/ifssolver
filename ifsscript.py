#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import asyncio
import sys
import time
from configparser import ConfigParser
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

import PortalTools
import SplitTools


def create_config(filename):
    config = {}
    config_raw = ConfigParser()
    config_raw.read(filename, encoding='utf-8')
    config['cookies'] = config_raw.get('Crawl', 'COOKIES', raw=True)
    config['map_config'] = {}
    config['map_config']['lat'] = config_raw.getfloat('Crawl', 'LAT')
    config['map_config']['lng'] = config_raw.getfloat('Crawl', 'LNG')
    config['map_config']['rad'] = config_raw.getfloat('Crawl', 'RADIUS')
    config['portals_csv'] = config_raw.get('Crawl', 'PORTALS_CSV', raw=True)
    config['portals_image_dir'] = config_raw.get('Crawl', 'IMAGE_DIR', raw=True)
    config['solver_config'] = {}
    config['solver_config']['target'] = config_raw.get('Solver', 'TARGET', raw=True)
    config['solver_config']['y_start'] = config_raw.getint('Solver', 'Y_START')
    config['solver_config']['y_end'] = config_raw.getint('Solver', 'Y_END')
    config['solver_config']['output'] = config_raw.get('Solver', 'OUTPUT', raw=True)
    config['solver_config']['spilt_result'] = config_raw.get('Solver', 'SPILT_RESULT', raw=True)
    config['solver_config']['threshold_up'] = config_raw.getint('Solver', 'THRESHOLD_UP')
    config['solver_config']['threshold_down'] = config_raw.getint('Solver', 'THRESHOLD_DOWN')
    return config


class Canvas(object):
    char_size, border_size = 80, 10

    def __init__(self):
        self.__canvas: Image = None
        self.__draw: ImageDraw = None

    @classmethod
    def create(cls, map_):
        self = Canvas()
        self.__canvas = Image.new(mode='RGB', color='black',
                                size=((cls.char_size + cls.border_size * 2) * len(map_),
                                      cls.char_size + cls.border_size * 2))
        self.__draw = ImageDraw.Draw(self.__canvas)
        return self

    def drawGlyph(self, n, po_col):
        xy_list = [(po['lng'], po['lat']) for po in po_col if po is not None]
        xy = np.array(xy_list, dtype=np.float)
        xy_sub = xy - np.min(xy, axis=0)
        xy_local = np.around((self.char_size / np.max(xy_sub)) * xy_sub)
        xy_global = np.zeros_like(xy_local)
        xy_global[:, 0] = xy_local[:, 0] + ((2 * n) + 1) * self.border_size + n * self.char_size
        xy_global[:, 1] = self.char_size - xy_local[:, 1] + self.border_size
        line_list = xy_global.astype(np.int0).flatten().tolist()
        self.__draw.line(line_list, width=3, fill=(255, 255, 255))

    def drawUnderline(self, n):
        line_list = [(((2 * n) + 1) * self.border_size + n * self.char_size, self.char_size + self.border_size + 5),
                     (((2 * n) + 1) * self.border_size + (n + 1) * self.char_size, self.char_size + self.border_size + 5)]
        self.__draw.line(line_list, width=5, fill=(255, 255, 255))

    def save(self, filename):
        self.__canvas.save(filename)


def drawSpiltResult(filename, img_path, rect_list):
    img = cv.imread(img_path)
    for rect in rect_list:
        x, y, h, w = rect
        cv.rectangle(img, (x, y), (x+h-1, y+w-1), (0, 0, 255), thickness=1)
    cv.imwrite(filename, img)
    print(f'[!] 图像分割结果保存到 {filename}')


async def crop(img, rect):
    x, y, h, w = rect
    return await asyncio.to_thread(img.crop, (x, y, x + h - 1, y + w - 1))


async def solver(config):
    sc = config['solver_config']
    thresh = (sc['threshold_down'], sc['threshold_up'])
    rect_list = await asyncio.to_thread(SplitTools.getPhotoContours, sc['target'], sc['y_start'], sc['y_end'], thresh)
    await asyncio.to_thread(drawSpiltResult, sc['spilt_result'], sc['target'], rect_list)
    map_ = await asyncio.to_thread(SplitTools.getPhotoMap, rect_list)
    img = await asyncio.to_thread(Image.open, sc['target'])
    hash_db = await PortalTools.ImageHashDatabase.create(config['portals_csv'], config['portals_image_dir'])
    canvas = await asyncio.to_thread(Canvas.create, map_)

    for n, col in enumerate(map_):
        print(f'[!] 正在处理第{n+1}列图像中...')
        result = [await hash_db.match(await crop(img, rect_list[num])) for num in col]
        await asyncio.to_thread(canvas.drawGlyph, n, result)
        if None in result:
            await asyncio.to_thread(canvas.drawUnderline, n)
            row_notfound = ','.join(str(n) for n, r in enumerate(result) if r is None)
            print(f'[!!!] 第{n+1}列图像中第{row_notfound}个结果不存在')

    await asyncio.to_thread(canvas.save, sc['output'])
    print(f"[!] Passcode的图像保存到 {sc['output']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A script used to obtain passcode from special picture')
    parser.add_argument('--config', dest='config', metavar='filename', default='config.ini',
                        action='store', help='Configure file, default = \'config.ini\'', required=False)

    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('-u', '--update', help='Update portals metadata only', action='store_true')
    group1.add_argument('-d', '--download', help='Download portals data for metadata only', action='store_true')
    group1.add_argument('-b', '--both', help='Both update and download', action='store_true')

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('-s', '--solve', help='Get passcode from picture automatically', action='store_true')

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f'[!] configure file \'{str(config_path)}\' not found.')
        sys.exit()
    print(f'[!] 正在使用配置文件({str(config_path)})运行程序')
    config = create_config(str(config_path))

    # 选择Windows上的事件循环
    if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def update():
        print('[!] 开始更新元数据...')
        await PortalTools.crawl(config['cookies'], config['map_config'], config['portals_csv'])
        print('')


    async def download():
        print('[!] 开始下载Intel上的图像...')
        pd = PortalTools.PortalImageDownloader(config['portals_image_dir'])
        await pd.download_from_csv(config['portals_csv'])
        print('')

    if args.update:
        asyncio.run(update())

    if args.download:
        asyncio.run(download())

    if args.both:
        async def both():
            await update()
            await download()
            print('[!] 全部完成')
        asyncio.run(both())

    if args.solve:
        print('[!] 开始自动处理')
        start = time.time()
        asyncio.run(solver(config))
        print(f'[!] 自动处理完成，用时 {time.time() - start} s')