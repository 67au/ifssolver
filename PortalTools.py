#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import asyncio
import csv
from functools import partial
from pathlib import Path

import aiofiles
import httpx
import imagehash
import numpy as np
from PIL import Image
from tqdm.asyncio import tqdm

from ingressAPI import AsyncIntelMap, MapTiles


class PortalsCSV(object):

    @staticmethod
    def write_csv(portals_csv, poInfo_list):
        portals_csv = Path(portals_csv)
        portals_csv.touch(exist_ok=True)
        with open(str(portals_csv), 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'lat', 'lng', 'url', 'uid']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(poInfo_list)

    @staticmethod
    def read_csv(filename):
        if not Path(filename).exists():
            print(f'{filename} does not found.')
            return []
        with open(filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            portals_list = list(reader)
        return portals_list


async def crawl(cookies, map_config, portals_csv):
    client = AsyncIntelMap(cookies=cookies)
    if not client.getCookieStatus():
        return

    # 构造下载的地图区间
    dpl = 111000  # distance per lat
    dlat = 1.0 * map_config['rad'] / dpl
    dlng = 1.0 * map_config['rad'] / (dpl * np.cos(map_config['lat'] / 180))
    bbox = [map_config['lng'] - dlng, map_config['lat'] - dlat, map_config['lng'] + dlng, map_config['lat'] + dlat]

    tiles = MapTiles(bbox).tiles
    print(f'[!] Tiles : {tiles}')
    print(f'[!] Number of tiles in boundry are : {len(tiles)}')
    print(f'[!] 正在更新Portal元数据...')
    portals_list = [portal async for portal in PortalsIterator(client, tiles)]
    await asyncio.to_thread(PortalsCSV.write_csv, portals_csv, portals_list)
    print(f'[!] CSV({portals_csv}) 写入成功。')


def PortalParser(poID, poDetail):
    lat = poDetail[2] / 1e6
    lng = poDetail[3] / 1e6
    url = poDetail[7]
    name = poDetail[8]
    uid = poID
    return dict(name=name, lat=lat, lng=lng, url=url, uid=uid)


async def PortalsIterator(client, tiles):
    async with client:
        tasks = [asyncio.create_task(getTileData(client, tile)) for tile in tiles]
        for coro in tqdm.as_completed(tasks):
            tile_data = await coro
            if tile_data is None:
                continue
            for _, val in tile_data['result']['map'].items():
                for entity in val['gameEntities']:
                    if entity[2][0] == 'p':
                        yield PortalParser(entity[0], entity[2])


async def getTileData(client: AsyncIntelMap, tile, zoom=15):
    iitc_xtile, iitc_ytile = (int(i) for i in tile)
    iitcTileName = f'{zoom}_{iitc_xtile}_{iitc_ytile}_0_8_100'
    data = await client.getEntites([iitcTileName])
    return data


class PortalImageDownloader(object):
    headers = {
        'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36'
    }
    limits = httpx.Limits(max_keepalive_connections=12, max_connections=30)

    def __init__(self, dirname):
        self.img_dir = Path(dirname)
        self.img_dir.mkdir(exist_ok=True)

    async def __download_img(self, client, portal):
        img_path = self.img_dir.joinpath(f"{portal['lng']}_{portal['lat']}.jpg")
        if img_path.exists():
            return True
        try:
            async with client.stream('GET', portal['url']) as r:
                if r.status_code == httpx.codes.OK:
                    async with aiofiles.open(str(img_path), 'wb') as f:
                        async for chunk in r.aiter_bytes():
                            await f.write(chunk)
                    return True
                else:
                    return False
        except:
            await asyncio.to_thread(partial(img_path.unlink, missing_ok=True))
            return False

    async def download_from_csv(self, portals_csv):
        portals_list = await asyncio.to_thread(PortalsCSV.read_csv, portals_csv)
        async with httpx.AsyncClient(headers=self.headers, limits=self.limits) as client:
            print(f'[!] 正在根据({portals_csv})下载Portal图像...')
            tasks = [asyncio.create_task(self.__download_img(client, portal)) for portal in portals_list]
            unfinished = [n for n, coro in enumerate(tqdm.as_completed(tasks), 1) if await coro is False]
            print(f'[!] 图像下载完成。')
            if any(unfinished):
                print(f"[!] 有{len(unfinished)}个图像下载失败，请重新执行下载剩余图像。")


class ImageHashDatabase(object):
    hash_func = {
        'ahash': imagehash.average_hash,
        'phash': imagehash.phash,
        'dhash': imagehash.dhash,
        'crop_resistant_hash': imagehash.crop_resistant_hash
    }

    def __init__(self):
        self.hash_method = None
        self.img_dir = None
        self.polist = None
        self.hash_list = None

    @classmethod
    async def create(cls, portals_csv, img_dir, hash_method='phash'):
        self = ImageHashDatabase()
        self.hash_method = cls.hash_func.get(hash_method, None)
        if self.hash_method is None:
            print(f'[!] 图像哈希方法({hash_method})不存在，退出。')
            exit()
        if not Path(portals_csv).exists():
            print(f'[!] 图像索引({portals_csv})不存在，退出。')
            exit()
        if not Path(img_dir).exists():
            print(f'[!] 图像目录({img_dir})不存在，退出。')
            exit()
        self.polist = await asyncio.to_thread(PortalsCSV.read_csv, portals_csv)
        self.img_dir = img_dir
        self.hash_list = await self.__init_db()
        return self

    async def getHash(self, img):
        return await asyncio.to_thread(self.hash_method, img)

    async def __getPortalHash(self, n, po):
        img_path = Path(self.img_dir).joinpath(f"{po['lng']}_{po['lat']}.jpg")
        if not img_path.exists():
            return None
        img = await asyncio.to_thread(Image.open, str(img_path))
        try:
            return n, await self.getHash(img)
        except:
            img.close()
            await asyncio.to_thread(img_path.unlink)
            print(f"[!!!] 请重新下载 {po['lng']}_{po['lat']}.jpg (Po名: {po['name']})")
            return None

    async def __init_db(self):
        print(f'[!] 正在构建imagehash数据库...')
        tasks = [asyncio.create_task(self.__getPortalHash(n, po)) for n, po in enumerate(self.polist)]
        hash_list = [result for coro in tqdm.as_completed(tasks) if (result := await coro) is not None]
        print(f'[!] 数据库构建完成，包含{len(hash_list)}个图像imagehash结果。')
        return hash_list

    async def match(self, img):
        hash_ = await self.getHash(img)
        sub = lambda n, h1, h2: (n, h1 - h2)
        tasks = [asyncio.create_task(asyncio.to_thread(sub, n, h, hash_)) for n, h in self.hash_list]
        result = [r for coro in asyncio.as_completed(tasks) if (r:=await coro)[1] <= 10]
        return self.polist[min(result, key=lambda r: r[1])[0]] if any(result) else None
