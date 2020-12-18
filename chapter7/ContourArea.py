# -*- coding:utf-8 -*-
import cv2 as cv
import sys
import numpy as np


if __name__ == '__main__':
    # 用四个点表示三角形轮廓
    A = (0, 0)              # 顶点A
    B = (10, 0)             # 顶点B
    C = (10, 10)            # 顶点C
    D = (5, 5)              # 斜边中点D
    triangle = np.array((A, B, C, D))
    triangle_area = cv.contourArea(triangle)
    print('三角形面积为：{}'.format(triangle_area))

    # 读取图像circles.png
    image = cv.imread('./images/circles.png')
    if image is None:
        print('Failed to read circles.png.')
        sys.exit()

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 高斯滤波
    gray = cv.GaussianBlur(gray, (9, 9), sigmaX=2, sigmaY=2)

    # 二值化
    binary = cv.threshold(gray, 75, 180, cv.THRESH_BINARY)

    # 轮廓检测
    contours, hierarchy = cv.findContours(binary[1], mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

    # 输出轮廓面积
    for i in range(len(contours)):
        img_area = cv.contourArea(contours[i])
        print('第{}个轮廓面积为：{}'.format(i, img_area))
