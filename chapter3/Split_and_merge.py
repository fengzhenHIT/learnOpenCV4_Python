# -*- coding:utf-8 -*-
import cv2 as cv
import sys
import numpy as np


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()

    # 通道分离
    b, g, r = cv.split(img)

    # 创建一个和图像尺寸相同的全0矩阵
    zeros = np.zeros(img.shape[:2], dtype='uint8')

    # 将通道数目相同的图像矩阵合并
    bg = cv.merge([b, g, zeros])
    gr = cv.merge([zeros, g, r])
    br = cv.merge([b, zeros, r])
    # 将通道数目不相同的图像矩阵合并
    bgr_6 = cv.merge([bg, r, zeros, zeros])

    # 展示结果
    cv.imshow('Blue', b)
    cv.imshow('Green', g)
    cv.imshow('Red', r)
    cv.imshow('Blue_Green', bg)
    cv.imshow('Green_Red', gr)
    cv.imshow('Blue_Red', br)

    # 关闭窗口
    cv.waitKey(0)
    cv.destroyAllWindows()
