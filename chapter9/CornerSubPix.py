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

    # 转为灰度图像，并将数据类型转为float32
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 检测Shi-tomasi角点
    corners = cv.goodFeaturesToTrack(gray, 100, 0.01, 0.04)
    corners1 = np.int0(corners)

    # 对角点进行备份
    corners2 = corners.copy()

    # 计算亚像素级别角点坐标
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 40, 0.001)
    corners2 = cv.cornerSubPix(gray, corners2, (5, 5), (-1, -1), criteria)

    # 输出初始坐标和精细坐标
    for i in range(len(corners)):
        print('第{}个角点的初始坐标为：{}，精细坐标为：{}'.format(i, corners1[i], corners2[i]))
