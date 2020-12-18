# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()
    # 沿x轴对称
    img_x = cv.flip(img, 0)
    # 沿y轴对称
    img_y = cv.flip(img, 1)
    # 先x轴对称，再y轴对称
    img_xy = cv.flip(img, -1)
    # 展示结果
    cv.imshow('img', img)
    cv.imshow('img_x', img_x)
    cv.imshow('img_y', img_y)
    cv.imshow('img_xy', img_xy)
    cv.waitKey(0)
    cv.destroyAllWindows()
