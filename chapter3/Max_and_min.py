# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 新建矩阵a和b
    a = np.array([1, 2, 3.3, 4, 5, 9, 5, 7, 8.2, 9, 10, 2])
    b = np.array([1, 2.2, 3, 1, 3, 10, 6, 7, 8, 9.3, 10, 1])
    img1 = np.reshape(a, (3, 4))
    img2 = np.reshape(b, (3, 4))
    img3 = np.reshape(a, (2, 3, 2))
    img4 = np.reshape(b, (2, 3, 2))

    # 对两个单通道图像矩阵进行比较运算
    max12 = cv.max(img1, img2)
    min12 = cv.min(img1, img2)

    # 对两个多通道图像矩阵进行比较运算
    max34 = cv.max(img3, img4)
    min34 = cv.min(img3, img4)

    # 对两张彩色图像进行比较运算
    img5 = cv.imread('./images/lena.jpg')
    img6 = cv.imread('./images/noobcv.jpg')
    max56 = cv.max(img5, img6)
    min56 = cv.min(img5, img6)
    cv.imshow('conMax', max56)
    cv.imshow('conMin', min56)

    # 对两张灰度图像进行比较运算
    img7 = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)
    img8 = cv.cvtColor(img6, cv.COLOR_BGR2GRAY)
    max78 = cv.max(img7, img8)
    min78 = cv.min(img7, img8)
    cv.imshow('conMax_GRAY', max78)
    cv.imshow('conMin_GRAY', min78)

    # 与掩模进行比较运算
    # 生成一个低通300*300的掩模矩阵
    src = np.zeros((512, 512, 3), dtype='uint8')
    src[100:400:, 100:400:] = 255
    min_img5_src = cv.min(img5, src)
    cv.imshow('Min img5 src', min_img5_src)

    # 生成一个显示红色通道的低通掩模矩阵
    src1 = np.zeros((512, 512, 3), dtype='uint8')
    src1[:, :, 2] = 255
    min_img5_src1 = cv.min(img5, src1)
    cv.imshow('Min img5 src1', min_img5_src1)

    # 关闭窗口
    cv.waitKey(0)
    cv.destroyAllWindows()
