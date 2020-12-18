# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 设置一个 2 维坐标和一个 3 维坐标
    point1 = np.array([[3, 6, 1.5]])
    point2 = np.array([[23, 32, 1]])

    # 非齐次坐标转齐次坐标
    point3 = cv.convertPointsToHomogeneous(point1)
    point4 = cv.convertPointsToHomogeneous(point2)

    # 齐次坐标转非齐次坐标
    point5 = cv.convertPointsFromHomogeneous(point1)
    point6 = cv.convertPointsFromHomogeneous(point2)

    # 输出结果
    print('非齐次坐标：{:<20}转为齐次坐标：{}'.format(str(point1), str(point3)))
    print('非齐次坐标：{:<20}转为齐次坐标：{}'.format(str(point2), str(point4)))

    print('齐次坐标：{:<22}转为非齐次坐标：{}'.format(str(point1), str(point5)))
    print('齐次坐标：{:<22}转为非齐次坐标：{}'.format(str(point2), str(point6)))
