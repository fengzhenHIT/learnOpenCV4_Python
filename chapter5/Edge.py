# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 读取图像equalLena.png
    image = cv.imread('./images/equalLena.png', cv.IMREAD_ANYCOLOR)
    if image is None:
        print('Failed to read equalLena.png.')
        sys.exit()

    # 创建边缘检测滤波器
    kernel1 = np.array([1, -1])
    kernel2 = np.array([1, 0, -1])
    kernel3 = kernel2.reshape((3, 1))
    kernel4 = np.array([1, 0, 0, -1]).reshape((2, 2))
    kernel5 = np.array([0, -1, 1, 0]).reshape((2, 2))

    # 检测图像边缘
    # 以[1, -1]检测水平方向边缘
    res1 = cv.filter2D(image, cv.CV_16S, kernel1)
    res1 = cv.convertScaleAbs(res1)
    # 以[1, 0, -1]检测水平方向边缘
    res2 = cv.filter2D(image, cv.CV_16S, kernel2)
    res2 = cv.convertScaleAbs(res2)
    # 以[1, 0, -1]检测垂直方向边缘
    res3 = cv.filter2D(image, cv.CV_16S, kernel3)
    res3 = cv.convertScaleAbs(res3)
    # 整幅图像边缘
    res = res2 + res3
    # 检测由左上到右下方向边缘
    res4 = cv.filter2D(image, cv.CV_16S, kernel4)
    res4 = cv.convertScaleAbs(res4)
    # 检测由右上到左下方向边缘
    res5 = cv.filter2D(image, cv.CV_16S, kernel5)
    res5 = cv.convertScaleAbs(res5)

    # 展示结果
    cv.imshow('Result1', res1)
    cv.imshow('Result2', res2)
    cv.imshow('Result3', res3)
    cv.imshow('Result', res)
    cv.imshow('Result4', res4)
    cv.imshow('Result5', res5)
    cv.waitKey(0)
    cv.destroyAllWindows()
