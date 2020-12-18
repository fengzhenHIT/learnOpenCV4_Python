# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像equalLena.png
    image = cv.imread('./images/equalLena.png', cv.IMREAD_ANYDEPTH)
    if image is None:
        print('Failed to read equalLena.png.')
        sys.exit()

    # X方向一阶边缘
    result_X = cv.Scharr(image, cv.CV_16S, 1, 0)
    result_X = cv.convertScaleAbs(result_X)
    # Y方向一阶边缘
    result_Y = cv.Scharr(image, cv.CV_16S, 0, 1)
    result_Y = cv.convertScaleAbs(result_Y)
    # 整幅图像的一阶边缘
    result_XY = result_X + result_Y

    # 显示结果
    cv.imshow('Result_X', result_X)
    cv.imshow('Result_Y', result_Y)
    cv.imshow('Result_XY', result_XY)
    cv.waitKey(0)
    cv.destroyAllWindows()
