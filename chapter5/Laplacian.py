# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像equalLena.png
    image = cv.imread('./images/equalLena.png', cv.IMREAD_ANYDEPTH)
    if image is None:
        print('Failed to read equalLena.png.')
        sys.exit()

    # 未滤波提取图像边缘
    result = cv.Laplacian(image, cv.CV_16S, ksize=3, scale=1, delta=0)
    result = cv.convertScaleAbs(result)
    # 滤波后提取图像边缘
    result_gauss = cv.GaussianBlur(image, (3, 3), 5, 0)
    result_gauss = cv.Laplacian(result_gauss, cv.CV_16S, ksize=3, scale=1, delta=0)
    result_gauss = cv.convertScaleAbs(result_gauss)

    # 显示结果
    cv.imshow('Result', result)
    cv.imshow('Result_Gauss', result_gauss)
    cv.waitKey(0)
    cv.destroyAllWindows()
