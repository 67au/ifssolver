#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cv2 as cv
import numpy as np
import portion as P


def getPhotoContours(img_path, y_start=0, y_end=0, thresh=200):
    """
    分割图像获取矩形轮廓（分割方法略为魔法，能用就行）
    :param img_path: 输入分割图像的路径
    :param y_start: 裁剪y轴方向的开头x像素
    :param y_end: 裁剪y轴方向的结尾x像素
    :param thresh: 像素与图像背景(50, 50, 50)的欧氏距离的平方（对于95%图像来说为200）
    :return: 分割获得的矩形列表
    """
    img = cv.imread(img_path)
    img_crop = img[y_start:img.shape[0] - y_end, :]
    std = np.array([50, 50, 50])
    img_temp = np.sum(np.power(img_crop - std, 2), axis=2)
    img_threshold = cv.inRange(img_temp, 0, thresh)
    kernel = np.ones((3, 3), np.uint8)
    edge = img_threshold
    img_zero = np.zeros_like(img_threshold)
    for i in range(10):
        edge = cv.morphologyEx(edge, cv.MORPH_OPEN, kernel=kernel, iterations=1)
        edge = cv.morphologyEx(edge, cv.MORPH_CLOSE, kernel=kernel, iterations=1)
        contours, _ = cv.findContours(255 - edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv.drawContours(img_zero, [cv.convexHull(cnt)], -1, 255, -1)
        edge = 255 - img_zero
    contours, _ = cv.findContours(255 - edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_zero2 = np.zeros_like(img_threshold)
    for cnt in contours:
        x, y, h, w = cv.boundingRect(cnt)
        cv.rectangle(img_zero2, (x, y), (x + h, y + w), 255, -1)
    contours, _ = cv.findContours(img_zero2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, h, w = cv.boundingRect(cnt)
        cv.rectangle(img_zero2, (x, y), (x + h, y + w), 255, -1)
    contours, _ = cv.findContours(img_zero2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE, offset=(0, y_start))
    rect_list = [cv.boundingRect(cnt) for cnt in contours]
    return rect_list


def getPhotoMap(rect_list):
    """
    将矩形轮廓排序
    :param rect_list: 输入矩形轮廓列表
    :return: 排序结果的二维数组
    """
    photo_col = []
    for n, cnt in enumerate(rect_list):
        flag = True
        x, _, w, _ = rect_list[n]
        for i in photo_col:
            inter = P.closed(x, x + w) & i['x_range']
            if not inter.empty:
                i['x_range'] = inter
                i['nums'].append(n)
                flag = False
                break
        if flag:
            photo_col.append(dict(x_range=P.closed(x, x + w), nums=[n]))

    photo_map = [sorted(col['nums'], key=lambda n: rect_list[n][1])
                 for col in sorted(photo_col, key=lambda c: c['x_range'][0])]
    return photo_map
