# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # LUT查找表第一层
    LUT_1 = np.zeros(256, dtype='uint8')
    LUT_1[101: 201] = 100
    LUT_1[201:] = 255
    # LUT查找表第二层
    LUT_2 = np.zeros(256, dtype='uint8')
    LUT_2[101: 151] = 100
    LUT_2[151: 201] = 150
    LUT_2[201:] = 255
    # LUT查找表第三层
    LUT_3 = np.zeros(256, dtype='uint8')
    LUT_3[0: 101] = 100
    LUT_3[101: 201] = 200
    LUT_3[201:] = 255

    # LUT三通道合并
    LUT = cv.merge((LUT_1, LUT_2, LUT_3))
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    out0 = cv.LUT(gray, LUT_1)
    out1 = cv.LUT(img, LUT_1)
    out2 = cv.LUT(img, LUT)

    # 展示结果
    cv.imshow('out0', out0)
    cv.imshow('out1', out1)
    cv.imshow('out2', out2)
    cv.waitKey(0)
    cv.destroyAllWindows()
