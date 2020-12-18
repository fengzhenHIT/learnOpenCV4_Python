# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 验证滤波算法的数据矩阵
    data = np.array([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25]], dtype='float32')

    # 构建X方向、Y方向和联合滤波器
    a = np.array([[-1], [3], [-1]])
    b = a.reshape((1, 3))
    ab = a * b

    # 验证高斯滤波的可分离性
    gaussX = cv.getGaussianKernel(3, 1)
    gauss_data = cv.GaussianBlur(data, (3, 3), 1, None, 1, cv.BORDER_CONSTANT)
    gauss_data_XY = cv.sepFilter2D(data, -1, gaussX, gaussX, None, (-1, -1), 0, cv.BORDER_CONSTANT)
    print('采用cv.GaussianBlur方式：\n{}'.format(gauss_data))
    print('采用cv.sepFilter2D方式：\n{}'.format(gauss_data_XY))

    # 线性滤波的可分离性
    data_Y = cv.filter2D(data, -1, a, None, (-1, -1), 0, cv.BORDER_CONSTANT)
    data_YX = cv.filter2D(data_Y, -1, b, None, (-1, -1), 0, cv.BORDER_CONSTANT)
    data_XY = cv.filter2D(data, -1, ab, None, (-1, -1), 0, cv.BORDER_CONSTANT)
    data_XY_sep = cv.sepFilter2D(data, -1, b, b, None, (-1, -1), 0, cv.BORDER_CONSTANT)
    print('data_Y=\n{}'.format(data_Y))
    print('data_YX=\n{}'.format(data_YX))
    print('data_XY=\n{}'.format(data_XY))
    print('data_XY_sep=\n{}'.format(data_XY_sep))

    # 对图像进行分离操作
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()

    img_Y = cv.filter2D(img, -1, a, None, (-1, -1), 0, cv.BORDER_CONSTANT)
    img_YX = cv.filter2D(img_Y, -1, b, None, (-1, -1), 0, cv.BORDER_CONSTANT)
    img_XY = cv.filter2D(img, -1, ab, None, (-1, -1), 0, cv.BORDER_CONSTANT)

    # 展示结果
    cv.imshow('Origin', img)
    cv.imshow('img Y', img_Y)
    cv.imshow('img YX', img_YX)
    cv.imshow('img XY', img_XY)
    cv.waitKey(0)
    cv.destroyAllWindows()
