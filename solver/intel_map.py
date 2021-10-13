import asyncio
import csv
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator, List

import aiofiles
import httpcore
import httpx
from httpx_socks import AsyncProxyTransport
from tqdm.asyncio import tqdm

from intelmapclient.IntelMapClient import AsyncClient, AsyncAPI
from intelmapclient.IntelMapClient.types import Portal
from intelmapclient.IntelMapClient.utils import MapTiles

from .types import PathType
from .utils import parse_portal_filename


class PortalUtils:

    @classmethod
    @asynccontextmanager
    async def get_intel_client(cls, cookies: str, proxy: str = None) -> 'AsyncClient':
        client = await AsyncClient.create_client(cookies, proxy)
        try:
            yield client
        finally:
            await client.close()

    @classmethod
    async def get_portals_from_square(cls,
                                      client: 'AsyncClient',
                                      center_lat: float,
                                      center_lng: float,
                                      radian_meter: int,
                                      ) -> Iterator['Portal']:
        map_tiles = MapTiles.from_square(
            center_lat=center_lat,
            center_lng=center_lng,
            radian_meter=radian_meter,
            zoom=15,
        )
        tile_container = await AsyncAPI.getEntitiesByTiles(client, map_tiles)
        return tile_container.portals()

    @classmethod
    async def _fetch_image_url(cls,
                               semaphore: 'asyncio.Semaphore',
                               client: 'httpx.AsyncClient',
                               filename: PathType,
                               url: str,
                               num: int,
                               max_tries: int = 20):
        async with semaphore:
            for _ in range(max_tries):
                try:
                    resp = await client.get(url)
                    async with aiofiles.open(filename, 'wb') as f:
                        await f.write(resp.content)
                    return True, num
                except (httpcore.ReadTimeout, httpx.ReadTimeout):
                    await asyncio.sleep(3)
            return False, num

    @classmethod
    async def download_portals_by_csv(cls, source_csv: PathType, target_dir: PathType,
                                      proxy: str = None, no_clean=False, max_worker: int = 10):
        loop = asyncio.get_event_loop()
        portal_list = await loop.run_in_executor(None, cls.read_portals_from_csv, source_csv)
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)
        semaphore = asyncio.Semaphore(max_worker) if proxy is None else asyncio.Semaphore(1)
        async with httpx.AsyncClient(
                headers={
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/93.0.4577.82 Safari/537.36'},
                transport=httpx.AsyncHTTPTransport(retries=1) if proxy is None else
                AsyncProxyTransport.from_url(proxy, retries=1)) as client:
            tasks = [
                asyncio.create_task(
                    cls._fetch_image_url(semaphore, client, target_dir.joinpath(parse_portal_filename(
                        lat=p['lat'], lng=p['lng'], image_url=p['image']
                    )), p['image'], n)
                ) for n, p in enumerate(portal_list)
                if not (no_clean and target_dir.joinpath(parse_portal_filename(
                        lat=p['lat'], lng=p['lng'], image_url=p['image']
                    )).exists())
            ]
            error_list = []
            for task in tqdm.as_completed(tasks):
                r, n = await task
                if not r:
                    error_list.append(n)
            if any(error_list):
                raise

    @classmethod
    def save_portals_as_csv(cls, filename: PathType, portals: Iterator['Portal']):
        Path(filename).parent.mkdir(exist_ok=True)
        with open(filename, 'w', newline='', encoding='utf8') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(['title', 'lat', 'lng', 'guid', 'image'])
            f_csv.writerows((p.title, p.latE6 / 1e6, p.lngE6 / 1e6, p.guid, p.image) for p in portals)

    @classmethod
    def read_portals_from_csv(cls, filename: PathType) -> List[dict]:
        with open(filename, 'r', encoding='utf8') as f:
            f_csv = csv.DictReader(f)
            return [i for i in f_csv]
