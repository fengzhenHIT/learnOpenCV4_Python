# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys

np.set_printoptions(suppress=True)


if __name__ == '__main__':
    # 创建一个数据类型为float64、维度为16*16的全1矩阵
    img = np.ones((16, 16), dtype='float64')

    # 在图像中加入随机噪声
    pts1 = np.random.rand(16, 16) - 0.5
    img += pts1

    # 计算标准求和积分
    sum1 = cv.integral(img)
    # 计算平方求和积分
    sum2, sqsum2 = cv.integral2(img)
    # 计算倾斜求和积分
    sum3, sqsum3, tilted3 = cv.integral3(img)

    # 展示结果
    cv.namedWindow('sum', cv.WINDOW_NORMAL)
    cv.namedWindow('sum_sqsum', cv.WINDOW_NORMAL)
    cv.namedWindow('sum_sqsum_tilted', cv.WINDOW_NORMAL)
    cv.imshow('sum', (sum1 / 255))
    cv.imshow('sum_sqsum', (sqsum2 / 255))
    cv.imshow('sum_sqsum_tilted', (tilted3 / 255))
    cv.waitKey(0)
    cv.destroyAllWindows()
