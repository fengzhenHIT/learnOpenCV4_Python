# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像approx.png
    image = cv.imread('./images/approx.png')
    if image is None:
        print('Failed to read approx.png.')
        sys.exit()

    # 灰度化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 二值化
    _, binary = cv.threshold(gray, 105, 255, cv.THRESH_BINARY)

    # 对图像进行开运算
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9), (-1, -1))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

    # 轮廓检测
    contours, hierarchy = cv.findContours(binary, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # 计算图像矩
    for i in contours:
        M = cv.moments(i)
        print('Spatial moments:')
        print('m00: {}, m10: {}, m01: {}, m20: {}, m11: {}, m02: {}, m30: {}, m21: {}, m12: {}, m03: {}'
              .format(M['m00'], M['m10'], M['m01'], M['m20'], M['m11'], M['m02'], M['m30'], M['m21'], M['m12'], M['m03']))
        print('Central moments:')
        print('mu20: {}, mu11: {}, mu02: {}, mu30: {}, mu21: {}, mu12: {}, mu03: {}'
              .format(M['mu20'], M['mu11'], M['mu02'], M['mu30'], M['mu21'], M['mu12'], M['mu03']))
        print('Central normalized moments:')
        print('nu20: {}, nu11: {}, nu02: {}, nu30: {}, nu21: {}, nu12: {}, nu03: {}'
              .format(M['nu20'], M['nu11'], M['nu02'], M['nu30'], M['nu21'], M['nu12'], M['nu03']))
