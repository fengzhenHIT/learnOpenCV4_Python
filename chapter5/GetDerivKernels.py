# -*- coding:utf-8 -*-
import cv2 as cv


if __name__ == '__main__':
    # 一阶X方向Sobel算子
    sobel_x1, sobel_y1 = cv.getDerivKernels(1, 0, 3)
    sobel_X1 = sobel_y1 * sobel_x1.T
    print('一阶X方向Sobel算子：\n{}'.format(sobel_X1))

    # 二阶X方向Sobel算子
    sobel_x2, sobel_y2 = cv.getDerivKernels(2, 0, 5)
    sobel_X2 = sobel_y2 * sobel_x2.T
    print('二阶X方向Sobel算子：\n{}'.format(sobel_X2))

    # 三阶X方向Sobel算子
    sobel_x3, sobel_y3 = cv.getDerivKernels(3, 0, 7)
    sobel_X3 = sobel_y3 * sobel_x3.T
    print('三阶X方向Sobel算子：\n{}'.format(sobel_X3))

    # X方向Scharr算子
    scharr_x, scharr_y = cv.getDerivKernels(1, 0, cv.FILTER_SCHARR)
    scharr_X = scharr_y * scharr_x.T
    print('X方向Scharr算子：\n{}'.format(scharr_X))
