# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像equalLena.png
    image = cv.imread('./images/equalLena.png', cv.IMREAD_ANYDEPTH)
    if image is None:
        print('Failed to read equalLena.png.')
        sys.exit()

    # 大阈值检测图像边缘
    result_high = cv.Canny(image, 100, 200, apertureSize=3)
    # 小阈值检测图像边缘
    result_low = cv.Canny(image, 20, 40, apertureSize=3)
    # 高斯模糊后检测图像边缘
    result_gauss = cv.GaussianBlur(image, (3, 3), 5)
    result_gauss = cv.Canny(result_gauss, 100, 200, apertureSize=3)

    # 显示结果
    cv.imshow('Result_high', result_high)
    cv.imshow('Result_low', result_low)
    cv.imshow('Result_gauss', result_gauss)
    cv.waitKey(0)
    cv.destroyAllWindows()
