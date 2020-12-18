# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 生成待膨胀图像image
    image = np.array([[0, 0, 0, 0, 255, 0],
                      [0, 255, 255, 255, 255, 255],
                      [0, 255, 255, 255, 255, 0],
                      [0, 255, 255, 255, 255, 0],
                      [0, 255, 255, 255, 255, 0],
                      [0, 0, 0, 0, 0, 0]], dtype='uint8')
    # 分别读取黑背景和白背景图
    black = cv.imread('./images/LearnCV_black.png', cv.IMREAD_GRAYSCALE)
    if black is None:
        print('Failed to read LearnCV_black.png.')
        sys.exit()
    white = cv.imread('./images/LearnCV_white.png', cv.IMREAD_GRAYSCALE)
    if white is None:
        print('Failed to read LearnCV_white.png.')
        sys.exit()

    # 生成两种结构元素：structure1为矩形结构，structure2为十字结构
    structure1 = cv.getStructuringElement(0, (3, 3))
    structure2 = cv.getStructuringElement(1, (3, 3))

    # 对img1进行膨胀
    dilate_image = cv.dilate(image, structure2)
    # 分别对黑背景和白背景图像进行矩形结构和十字结构元素膨胀
    dilate_black_1 = cv.dilate(black, structure1)
    dilate_black_2 = cv.dilate(black, structure2)
    dilate_white_1 = cv.dilate(white, structure1)
    dilate_white_2 = cv.erode(white, structure2)
    # 比较膨胀和腐蚀的结果
    erode_black = cv.erode(black, structure1)
    result_xor = cv.bitwise_xor(erode_black, dilate_white_1)
    result_and = cv.bitwise_and(erode_black, dilate_white_1)

    # 展示结果
    cv.namedWindow('image', 0)
    cv.namedWindow('image dilate', 0)
    cv.imshow('image', image)
    cv.imshow('image dilate', dilate_image)
    cv.imshow('LearnCV black', black)
    cv.imshow('LearnCV black dilate structure1', dilate_black_1)
    cv.imshow('LearnCV black dilate structure2', dilate_black_2)
    cv.imshow('LearnCV white', white)
    cv.imshow('LearnCV white dilate structure1', dilate_white_1)
    cv.imshow('LearnCV white dilate structure2', dilate_white_2)
    cv.imshow('Result Xor', result_xor)
    cv.imshow('Result And', result_and)

    cv.waitKey(0)
    cv.destroyAllWindows()
