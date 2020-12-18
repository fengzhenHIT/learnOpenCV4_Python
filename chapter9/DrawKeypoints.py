# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 读取图像
    image = cv.imread('./images/lena.jpg')
    if image is None:
        print('Failed to read lena.jpg.')
        sys.exit()

    # 转为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 生成关键点
    kps = []
    key_points = np.random.randint(0, 512, 200).reshape((100, 2))
    for point in key_points:
        kps.append(cv.KeyPoint(point[0], point[1], 1))

    # 绘制关键点
    image_result = cv.drawKeypoints(image, kps, None, (), cv.DRAW_MATCHES_FLAGS_DEFAULT)
    gray_result = cv.drawKeypoints(gray, kps, None, (), cv.DRAW_MATCHES_FLAGS_DEFAULT)

    # 展示结果
    cv.imshow('Color Result', image_result)
    cv.imshow('Gray Result', gray_result)
    cv.waitKey(0)
    cv.destroyAllWindows()
