# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def compute_rvec(points1, points2, matrix, coeffs):
    # 用PnP算法计算旋转和平移向量
    _, rvec1, tvec1 = cv.solvePnP(points1, points2, matrix, coeffs)
    # 用PnP+Ransac算法计算旋转向量和平移向量
    _, rvec2, tvec2, inliers = cv.solvePnPRansac(points1, points2, matrix, coeffs)

    # 旋转向量转换为旋转矩阵
    rvec1_transport, _ = cv.Rodrigues(rvec1)
    rvec2_transport, _ = cv.Rodrigues(rvec2)

    # 输出结果
    print('世界坐标系变换到相机坐标系的旋转向量（cv.solvePnP）：\n', rvec1)
    print('对应旋转矩阵为：\n', rvec1_transport)
    print('世界坐标系变换到相机坐标系的旋转向量（cv.solvePnPRansac）：\n', rvec2)
    print('对应旋转矩阵为：\n', rvec2_transport)


if __name__ == '__main__':
    # 输入CalibrateCamera.py程序中得到的内参矩阵
    cameraMatrix = np.array([[532.01629758, 0, 332.17251924],
                             [0, 531.56515879, 233.38807482],
                             [0, 0, 1]])

    # 输入CalibrateCamera.py程序中得到的畸变系数
    distCoeffs = np.array([[-0.28518841, 0.08009721, 0.00127403, -0.00241511, 0.10657911]])

    # 生成棋盘格内角点的三维坐标
    obj_points = np.zeros((54, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    obj_points = np.reshape(obj_points, (54, 1, 3))

    # 读取图像
    img = cv.imread('./images/left04.jpg')
    if img is None:
        print('Failed to read left04.jpg.')
        sys.exit()
    # 转为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 定义放个标定板内角点数目（行，列）
    board_size = (9, 6)
    # 计算方格标定板角点
    _, points = cv.findChessboardCorners(gray, board_size)
    # 细化角点坐标
    _, points = cv.find4QuadCornerSubpix(gray, points, (5, 5))

    # 计算两个坐标系之间的旋转向量及旋转矩阵
    compute_rvec(obj_points, points, cameraMatrix, distCoeffs)

    # 修改其中一个三维坐标，重新进行计算
    obj_points[53] = [[8, 8, 0]]
    compute_rvec(obj_points, points, cameraMatrix, distCoeffs)
