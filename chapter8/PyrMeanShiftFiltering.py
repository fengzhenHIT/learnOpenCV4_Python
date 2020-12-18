# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 读取图像keys.jpg
    image = cv.imread('./images/keys.jpg')
    if image is None:
        print('Failed to read keys.jpg.')
        sys.exit()

    # 定义迭代算法终止条件
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    # 进行分割
    result1 = cv.pyrMeanShiftFiltering(image, 20, 40, maxLevel=2, termcrit=criteria)
    result2 = cv.pyrMeanShiftFiltering(result1, 20, 40, maxLevel=2, termcrit=criteria)

    # 对图像进行Canny边缘提取
    img_canny = cv.Canny(image, 150, 300)
    result1_canny = cv.Canny(result1, 150, 300)
    result2_canny = cv.Canny(result2, 150, 300)

    # 展示结果
    cv.imshow('Origin', image)
    cv.imshow('Origin Canny', img_canny)
    cv.imshow('Result1', result1)
    cv.imshow('Result1 Canny', result1_canny)
    cv.imshow('Result2', result2)
    cv.imshow('Result2 Canny', result2_canny)
    cv.waitKey(0)
    cv.destroyAllWindows()
