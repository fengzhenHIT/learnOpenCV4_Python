# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像
    image1 = cv.imread('./images/left01.jpg')
    image2 = cv.imread('./images/circle.png')
    if image1 is None or image2 is None:
        print('Failed to read left01.jpg or circle.png.')
        sys.exit()

    # 转为灰度图像
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    # 定义数目尺寸
    board_size1 = (9, 6)
    board_size2 = (7, 7)

    # 检测角点
    _, points1 = cv.findChessboardCorners(gray1, board_size1)
    _, points2 = cv.findCirclesGrid(gray2, board_size2)

    # 细化角点坐标
    _, points1 = cv.find4QuadCornerSubpix(gray1, points1, (5, 5))
    _, points2 = cv.find4QuadCornerSubpix(gray2, points2, (5, 5))

    # 绘制角点检测结果
    image1 = cv.drawChessboardCorners(image1, board_size1, points1, True)
    image2 = cv.drawChessboardCorners(image2, board_size2, points2, True)

    # 展示结果
    cv.imshow('Square Result', image1)
    cv.imshow('Circle Result', image2)
    cv.waitKey(0)
    cv.destroyAllWindows()
