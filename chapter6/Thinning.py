# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 对图像进行读取
    img1 = cv.imread('./images/LearnCV_black.png', cv.IMREAD_GRAYSCALE)
    if img1 is None:
        print('Failed to read LearnCV_black.png.')
        sys.exit()
    img2 = cv.imread('./images/OpenCV_4.1.png', cv.IMREAD_GRAYSCALE)
    if img2 is None:
        print('Failed to read OpenCV_4.1.png.')
        sys.exit()

    # 对图片进行细化
    thin1 = cv.ximgproc.thinning(img1, thinningType=0)
    thin2 = cv.ximgproc.thinning(img2, thinningType=0)

    # 展示结果
    cv.imshow('img1', img1)
    cv.imshow('img1_thinning', thin1)
    cv.imshow('img2', img2)
    cv.imshow('img2_thinning', thin2)

    cv.waitKey(0)
    cv.destroyAllWindows()
