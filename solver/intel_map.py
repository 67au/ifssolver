import asyncio
import csv
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import List, Tuple, Union

import aiofiles
import httpx
from httpx_socks import AsyncProxyTransport
from IntelMapClient import AsyncClient, AsyncAPI
from IntelMapClient.client import DEFAULT_HEADERS
from IntelMapClient.types import MapTiles, Portal
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .types import PathType
from .utils import parse_portal_filename

FIELD_NAMES = ['Name', 'Latitude', 'Longitude', 'Image']
MAX_WORKERS = 10


class PortalDownloader:

    def __init__(self,
                 image_dir: PathType,
                 proxy_url: str = None,
                 no_clean: bool = True,
                 max_workers: int = MAX_WORKERS
                 ):
        self.image_dir = Path(image_dir)
        self.proxy_url = proxy_url
        self.no_clean = no_clean
        self.max_workers = max_workers

        self.logger = logging.getLogger(__name__)

    async def iter_portals_by_square(self,
                                     cookies: str,
                                     center_lat: float,
                                     center_lng: float,
                                     radian_meter: int,
                                     ) -> Iterator[Portal]:
        map_tiles = MapTiles.from_square(
            center_lat=center_lat,
            center_lng=center_lng,
            radian_meter=radian_meter,
            zoom=15,
        )
        async with AsyncClient(cookies, self.proxy_url) as client:
            client.set_workers(self.max_workers)
            if not await client.authorize():
                self.logger.error('Cookies 验证失败')
                sys.exit(0)
            tile_set = await AsyncAPI(client).GetEntitiesByMapTiles(map_tiles)
            return tile_set.portals()

    async def _save_image(self,
                          image: bytes,
                          image_name,
                          ) -> None:
        async with aiofiles.open(self.image_dir.joinpath(image_name), 'wb') as f:
            await f.write(image)

    async def _fetch_image(self,
                           semaphore: asyncio.Semaphore,
                           client: httpx.AsyncClient,
                           url: str,
                           filename: PathType,
                           num: int,
                           ) -> Tuple[int, Union[str, Exception, None]]:
        async with semaphore:
            if not url.startswith('http://'):
                return num, 'Not URL'
            try:
                resp = await client.get(url)
                await self._save_image(resp.content, filename)
            except Exception as e:
                return num, e
            return num, None

    async def download_portals_by_list(self, portals_list: list) -> Tuple[bool, Union[list, None]]:
        semaphore = asyncio.Semaphore(self.max_workers)
        async with httpx.AsyncClient(
                headers=DEFAULT_HEADERS,
                transport=httpx.AsyncHTTPTransport(retries=1) if self.proxy_url is None else
                AsyncProxyTransport.from_url(self.proxy_url, retries=1),
                timeout=httpx.Timeout(15),
        ) as client:
            tasks = []
            for num, p in enumerate(portals_list):
                filename = parse_portal_filename(p['Image'], p['Latitude'], p['Longitude'])
                if self.no_clean and self.image_dir.joinpath(filename).exists():
                    continue
                tasks.append(
                    asyncio.create_task(self._fetch_image(
                        semaphore=semaphore,
                        client=client,
                        url=p['Image'],
                        filename=filename,
                        num=num,
                    )
                ))
            errors_list = []
            with logging_redirect_tqdm():
                for task in tqdm.as_completed(tasks):
                    num, error = await task
                    if error is not None:
                        errors_list.append((num, error))
            if any(errors_list):
                return False, errors_list
            else:
                return True, None

    @staticmethod
    def save_portals_as_csv(filename: PathType, portals: Iterator[Portal]) -> None:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(FIELD_NAMES)
            f_csv.writerows(((p.title, p.lat, p.lng, p.image) for p in portals))

    @staticmethod
    def read_portals_from_csv(filename: PathType) -> List[dict]:
        with open(filename, 'r', newline='', encoding='utf-8', errors="replace") as f:
            f_csv = csv.DictReader(f, fieldnames=FIELD_NAMES)
            _ = next(f_csv)
            return list(f_csv)  # type: ignore
