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

    # 创建ORB对象
    orb = cv.ORB_create(500, 1.2, 8, 31, 0, 2, cv.ORB_HARRIS_SCORE, 31, 20)

    # 计算ORB特征点
    kps = orb.detect(image, None)

    # 计算ORB描述子
    descriptions = orb.compute(image, kps)

    # 绘制ORB特征点
    image1 = image.copy()
    # 不含角度和大小
    image = cv.drawKeypoints(image, kps, image, ())
    # 包含角度和大小
    image1 = cv.drawKeypoints(image1, kps, image1, (), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 展示结果
    cv.imshow('ORB KeyPoints', image)
    cv.imshow('ORB KeyPoints(with Angle and Size)', image1)
    cv.waitKey(0)
    cv.destroyAllWindows()
