# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 矩阵的垂直和水平连接
    # 定义矩阵A和B
    A = np.array([[1, 7], [2, 8]])
    B = np.array([[4, 10], [5, 11]])
    # 垂直连接
    V_C = cv.vconcat((A, B))
    # 水平连接
    H_C = cv.hconcat((A, B))
    print('垂直连接结果：\n{}'.format(V_C))
    print('水平连接结果：\n{}'.format(H_C))

    # 图像的垂直和水平连接
    # 读取四张图像
    # 读取图像并判断是否读取成功
    img00 = cv.imread('./images/lena00.jpg')
    img01 = cv.imread('./images/lena01.jpg')
    img10 = cv.imread('./images/lena10.jpg')
    img11 = cv.imread('./images/lena11.jpg')
    if img00 is None or img01 is None or img10 is None or img11 is None:
        print('Failed to read images.')
        sys.exit()

    # 图像连接
    # 水平连接
    img0 = cv.hconcat((img00, img01))
    img1 = cv.hconcat((img10, img11))
    # 垂直连接
    img = cv.vconcat((img0, img1))
    # 显示结果
    cv.imshow('img00', img00)
    cv.imshow('img01', img01)
    cv.imshow('img10', img10)
    cv.imshow('img11', img11)
    cv.imshow('img0', img0)
    cv.imshow('img1', img1)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
