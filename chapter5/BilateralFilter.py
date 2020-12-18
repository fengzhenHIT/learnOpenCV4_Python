# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像face1.png和face2.png
    image1 = cv.imread('./images/face1.png', cv.IMREAD_ANYCOLOR)
    image2 = cv.imread('./images/face2.png', cv.IMREAD_ANYCOLOR)
    if image1 is None or image2 is None:
        print('Failed to read face1.png or face2.png.')
        sys.exit()

    # 验证不同滤波器直径的滤波效果
    res1 = cv.bilateralFilter(image1, 9, 50, 25 / 2)
    res2 = cv.bilateralFilter(image1, 25, 50, 25 / 2)

    # 验证不同标准差值的滤波效果
    res3 = cv.bilateralFilter(image2, 9, 9, 9)
    res4 = cv.bilateralFilter(image2, 9, 200, 200)

    # 展示结果
    cv.imshow('Origin_image1', image1)
    cv.imshow('Origin_image2', image2)
    cv.imshow('Result1', res1)
    cv.imshow('Result2', res2)
    cv.imshow('Result3', res3)
    cv.imshow('Result4', res4)

    cv.waitKey(0)
    cv.destroyAllWindows()
