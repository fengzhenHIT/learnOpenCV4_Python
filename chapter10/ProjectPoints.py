# -*- coding:utf-8 -*-
"""
本程序中用到的图像是代码清单10-10中相机标定时的第一张图像
各项参数都是标定时得到的结果
"""
import cv2 as cv
import numpy as np
import sys

if __name__ == '__main__':
    # 输入CalibrateCamera.py程序中得到的内参矩阵
    cameraMatrix = np.array([[532.01629758, 0, 332.17251924],
                             [0, 531.56515879, 233.38807482],
                             [0, 0, 1]])

    # 输入CalibrateCamera.py程序中得到的畸变系数
    distCoeffs = np.array([[-0.28518841, 0.08009721, 0.00127403, -0.00241511, 0.10657911]])

    # 输入CalibrateCamera.py程序中得到的相机坐标系与世界坐标系之间的旋转向量和平移向量
    rvecs = np.array([[0.16460723], [0.29404635], [0.01212824]])
    tvecs = np.array([[-2.6881551], [-4.27993647], [15.91970296]])

    # 读取图像
    img = cv.imread('./images/left01.jpg')
    if img is None:
        print('Failed to read left0{}.jpg.'.format(i))
        sys.exit()

    # 转为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 定义放个标定板内角点数目（行，列）
    board_size = (9, 6)
    # 计算方格标定板角点
    _, points = cv.findChessboardCorners(gray, board_size)
    # 细化角点坐标
    _, points = cv.find4QuadCornerSubpix(gray, points, (5, 5))

    # 生成棋盘格内角点的三维坐标
    obj_points = np.zeros((54, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    obj_points = np.reshape(obj_points, (54, 1, 3))

    # 根据三维坐标和相机与世界坐标系时间的关系估计内角点像素坐标
    points1, _ = cv.projectPoints(obj_points, rvecs, tvecs, cameraMatrix, distCoeffs)

    # 计算图像中内角点的真实坐标误差
    error = 0
    for j in range(len(points)):
        error += np.sqrt(np.power((points[j][0][0] - points1[j][0][0]), 2) + np.power((points[j][0][1] - points1[j][0][1]), 2))
    print('图像中内角点的真实坐标误差为：{}'.format(round(error / len(points), 6)))
