# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 创建矩阵，用于求像素之间的距离
    array = np.array([[1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 1, 0, 1, 1],
                      [1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1]], dtype='uint8')
    # 分别计算街区距离、欧氏距离和棋盘距离
    dst_L1 = cv.distanceTransform(array, cv.DIST_L1, cv.DIST_MASK_3)
    dst_L2 = cv.distanceTransform(array, cv.DIST_L2, cv.DIST_MASK_5)
    dst_C = cv.distanceTransform(array, cv.DIST_C, cv.DIST_MASK_3)

    # 对图像进行读取
    rice = cv.imread('./images/rice.png', cv.IMREAD_GRAYSCALE)
    if rice is None:
        print('Failed to read rice.png.')
        sys.exit()

    # 将图像转成二值图像，同时将黑白区域互换
    rice_BW = cv.threshold(rice, 50, 255, cv.THRESH_BINARY)
    rice_BW_INV = cv.threshold(rice, 50, 255, cv.THRESH_BINARY_INV)

    # 图像距离变换
    dst_rice_BW = cv.distanceTransform(rice_BW[1], 1, 3, dstType=cv.CV_32F)
    dst_rice_BW_INV = cv.distanceTransform(rice_BW_INV[1], 1, 3, dstType=cv.CV_8U)

    # 展示矩阵距离计算结果
    print('街区距离：\n{}'.format(dst_L1))
    print('欧氏距离：\n{}'.format(dst_L2))
    print('棋盘距离：\n{}'.format(dst_C))

    # 展示二值化、黑白互换后的图像及距离变换结果
    cv.imshow('rice_BW', rice_BW[1])
    cv.imshow('rice_BW_INV', rice_BW_INV[1])
    cv.imshow('dst_rice_BW', dst_rice_BW)
    cv.imshow('dst_rice_BW_INV', dst_rice_BW_INV)

    cv.waitKey(0)
    cv.destroyAllWindows()
