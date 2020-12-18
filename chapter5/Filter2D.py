# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 1. 以矩阵为例
    src = np.array([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25]], dtype='float32')
    kernel1 = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], dtype='float32') / 9
    result = cv.filter2D(src, -1, kernel=kernel1)
    print('卷积前矩阵：\n{}'.format(src))
    print('卷积后矩阵：\n{}'.format(result))

    # 2. 以图像为例
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()
    kernel2 = np.ones((7, 7), np.float32) / 49
    result2 = cv.filter2D(img, -1, kernel=kernel2)

    # 展示结果
    cv.imshow('Origin Image', img)
    cv.imshow('Filter Result', result2)
    cv.waitKey(0)
    cv.destroyAllWindows()
