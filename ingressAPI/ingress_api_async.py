#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import time

import httpx


class IntelMap(object):
    headers = {
        'accept': '*/*',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.8',
        'content-type': 'application/json; charset=UTF-8',
        'origin': 'https://intel.ingress.com',
        'referer': 'https://intel.ingress.com/intel',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36',
    }
    data_base = {
        'v': '',
    }
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)

    def __init__(self, cookies, proxies=None):
        self.__isCookieOK = False
        self.proxies = proxies or {}
        self.client = httpx.AsyncClient(headers=self.headers, limits=self.limits, proxies=self.proxies)
        self.__login(cookies=cookies)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def __login(self, cookies):
        try:
            cookies_dict = {k.strip(): v for k, v in re.findall(r'(.*?)=(.*?);', cookies)}
            client = httpx.Client(headers=self.headers, cookies=cookies_dict, proxies=self.proxies)
            test = client.get('https://intel.ingress.com/intel')
            self.data_base['v'] = re.findall('/jsc/gen_dashboard_([\d\w]+).js"', test.text)[0]
            print('cookies success')
            self.headers.update({'x-csrftoken': client.cookies.get('csrftoken', domain='intel.ingress.com')})
            self.client.headers = self.headers
            self.client.cookies = client.cookies
            self.__isCookieOK = True
        except IndexError:
            print("Oops!, looks like you have a problem with your cookie.")
            self.__isCookieOK = False

    def getCookieStatus(self):
        return self.__isCookieOK

    async def __request(self, url, data):
        data = json.dumps(data)
        for _ in range(10):
            try:
                # noinspection PyTypeChecker
                r = await self.client.post(url=url, data=data)
                return r.json()
            except json.JSONDecodeError:
                continue
            except httpx.ReadTimeout:
                continue
        return None

    async def getGamescore(self):
        data = self.data_base
        url = 'https://intel.ingress.com/r/getGameScore'
        return await self.__request(url=url, data=data)

    async def getEntites(self, tilenames):
        _ = {"tileKeys": tilenames}  # ['15_26070_13886_8_8_100']
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/getEntities'
        return await self.__request(url=url, data=data)

    async def getPortalDetails(self, guid):
        _ = {"guid": guid}  # 3e2bcc15c58d486fae24e2ade2bf7327.16
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/getPortalDetails'
        return await self.__request(url=url, data=data)

    async def getPlexts(self,
                        min_lng,
                        max_lng,
                        min_lat,
                        max_lat,
                        tab='all',
                        maxTimestampMs=-1,
                        minTimestampMs=0,
                        ascendingTimestampOrder=True):
        minTimestampMs = int(time.time() * 1000) if minTimestampMs == 0 else minTimestampMs
        _ = {
            'ascendingTimestampOrder': ascendingTimestampOrder,
            'maxLatE6': max_lat,
            'minLatE6': min_lat,
            'maxLngE6': max_lng,
            'minLngE6': min_lng,
            'maxTimestampMs': maxTimestampMs,
            'minTimestampMs': minTimestampMs,
            'tab': tab
        }
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/getPlexts'
        return await self.__request(url=url, data=data)

    async def sendPlexts(self, lat, lng, message, tab='faction'):
        _ = {
            'latE6': lat,
            'lngE6': lng,
            'message': message,
            'tab': tab
        }
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/sendPlext'
        return await self.__request(url=url, data=data)

    async def getRegionScoreDetails(self, lat, lng):
        _ = {
            'latE6': lat,  # 30420109, 104938641
            'lngE6': lng
        }
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/getRegionScoreDetails'
        return await self.__request(url=url, data=data)

    async def redeemReward(self, passcode):
        _ = {'passcode': passcode}
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/redeemReward'
        return await self.__request(url=url, data=data)