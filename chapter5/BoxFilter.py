# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/equalLena.png', cv.IMREAD_ANYDEPTH)
    if img is None:
        print('Failed to read equalLena.png.')
        sys.exit()

    # 验证方框滤波算法的数组矩阵
    points = np.array([[1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10],
                       [11, 12, 13, 14, 15],
                       [16, 17, 18, 19, 20],
                       [21, 22, 23, 24, 25]], dtype='float32')

    # 将图像转为float32类型的数据
    img_32 = img.astype('float32')
    img_32 /= 255.0

    # 方框滤波cv.boxFilter()和cv.sqrBoxFilter()
    # 进行归一化
    img_box_norm = cv.boxFilter(img, -1, (3, 3), anchor=(-1, -1), normalize=True)
    # 不进行归一化
    img_box = cv.boxFilter(img, -1, (3, 3), anchor=(-1, -1), normalize=False)

    # 进行归一化
    points_sqr_norm = cv.sqrBoxFilter(points, -1, (3, 3), anchor=(-1, -1),
                                      normalize=True, borderType=cv.BORDER_CONSTANT)
    img_sqr_norm = cv.sqrBoxFilter(img, -1, (3, 3), anchor=(-1, -1),
                                    normalize=True, borderType=cv.BORDER_CONSTANT)
    # 不进行归一化
    points_sqr = cv.sqrBoxFilter(points, -1, (3, 3), anchor=(-1, -1),
                                 normalize=False, borderType=cv.BORDER_CONSTANT)

    # 展示图像处理结果
    cv.imshow('Result(cv.boxFilter() NORM)', img_box_norm)
    cv.imshow('Result(cv.boxFilter()', img_box)
    cv.imshow('Result(cv.sqrBoxFilter() NORM', img_sqr_norm / np.max(img_sqr_norm))
    cv.waitKey(0)
    cv.destroyAllWindows()
