#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import re
import time

import httpx


def get_tiles_per_edge(zoom):
    if zoom > 15:
        zoom = 15
    elif zoom < 3:
        zoom = 3
    else:
        pass
    return [1, 1, 1, 40, 40, 80, 80, 320, 1000, 2000, 2000, 4000, 8000, 16000, 16000, 32000][zoom]


def lng2tile(lng, tpe):  # w
    return int((lng + 180) / 360 * tpe)


def lat2tile(lat, tpe):  # j
    return int((1 - math.log(math.tan(lat * math.pi / 180) + 1 / math.cos(lat * math.pi / 180)) / math.pi) / 2 * tpe)


def tile2lng(x, tpe):
    return x / tpe * 360 - 180


def tile2lat(y, tpe):
    n = math.pi - 2 * math.pi * y / tpe
    return 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))


class MapTiles(object):

    def __init__(self, bbox, zoom=15):
        self.LowerLng = bbox[0]
        self.LowerLat = bbox[1]
        self.UpperLng = bbox[2]
        self.UpperLat = bbox[3]
        self.zpe = get_tiles_per_edge(zoom)
        self.tiles = self.__initTiles()

    def __initTiles(self):
        Lx = lng2tile(self.LowerLng, self.zpe)
        Ly = lat2tile(self.LowerLat, self.zpe)
        Ux = lng2tile(self.UpperLng, self.zpe)
        Uy = lat2tile(self.UpperLat, self.zpe)
        tiles = list([x, y] for x in range(Lx, Ux + 1) for y in range(Uy, Ly + 1))
        return tiles


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

    def __init__(self, cookies, proxies=None):
        self.__isCookieOK = False
        self.proxies = proxies or {}
        self.client = httpx.Client(headers=self.headers, proxies=self.proxies)
        self.__login(cookies=cookies)

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

    def __request(self, url, data):
        data = json.dumps(data)
        for _ in range(10):
            try:
                # noinspection PyTypeChecker
                r = self.client.post(url=url, data=data)
                return r.json()
            except json.JSONDecodeError:
                continue
        return None

    def getGamescore(self):
        data = self.data_base
        url = 'https://intel.ingress.com/r/getGameScore'
        return self.__request(url=url, data=data)

    def getEntites(self, tilenames):
        _ = {"tileKeys": [tilenames]}  # ['15_26070_13886_8_8_100']
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/getEntities'
        return self.__request(url=url, data=data)

    def getPortalDetails(self, guid):
        _ = {"guid": guid}  # 3e2bcc15c58d486fae24e2ade2bf7327.16
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/getPortalDetails'
        return self.__request(url=url, data=data)

    def getPlexts(self,
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
        return self.__request(url=url, data=data)

    def sendPlexts(self, lat, lng, message, tab='faction'):
        _ = {
            'latE6': lat,
            'lngE6': lng,
            'message': message,
            'tab': tab
        }
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/sendPlext'
        return self.__request(url=url, data=data)

    def getRegionScoreDetails(self, lat, lng):
        _ = {
            'latE6': lat,  # 30420109, 104938641
            'lngE6': lng
        }
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/getRegionScoreDetails'
        return self.__request(url=url, data=data)

    def redeemReward(self, passcode):
        _ = {'passcode': passcode}
        data = dict(self.data_base, **_)
        url = 'https://intel.ingress.com/r/redeemReward'
        return self.__request(url=url, data=data)