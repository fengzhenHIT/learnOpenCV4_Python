# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def compute_points(img):
    # 转为灰度图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 定义方格标定板内角点数目（行，列）
    board_size = (9, 6)
    # 计算方格标定板角点
    _, points = cv.findChessboardCorners(gray, board_size)
    # 细化角点坐标
    _, points = cv.find4QuadCornerSubpix(gray, points, (5, 5))
    return points


if __name__ == '__main__':
    # 生成棋盘格内角点的三维坐标
    obj_points = np.zeros((54, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    obj_points = np.reshape(obj_points, (54, 1, 3)) * 10

    # 计算棋盘格内角点的三维坐标及其在图像中的二维坐标
    all_obj_points = []
    all_points_L = []
    all_points_R = []
    for i in range(1, 5):
        # 读取图像
        imageL = cv.imread('./images/left0{}.jpg'.format(i))
        if imageL is None:
            print('Failed to read left0{}.jpg.'.format(i))
            sys.exit()
        imageR = cv.imread('./images/right0{}.jpg'.format(i))
        if imageR is None:
            print('Failed to read right0{}.jpg.'.format(i))
            sys.exit()

        # 获取图像尺寸
        h, w = image.shape[:2]
        # 计算三维坐标
        all_obj_points.append(obj_points)
        # 计算二维坐标
        all_points_L.append(compute_points(imageL))
        all_points_R.append(compute_points(imageR))

    # 分别计算相机内参矩阵和畸变系数
    _, cameraMatrix1, distCoeffs1, rvecs1, tvecs1 = cv.calibrateCamera(all_obj_points, all_points_L, (w, h), None, None)
    _, cameraMatrix2, distCoeffs2, rvecs2, tvecs2 = cv.calibrateCamera(all_obj_points, all_points_R, img_size, None, None)

    # 进行标定
    _, _, _, _, _, R, T, E, F = cv.stereoCalibrate(all_obj_points, all_points_L, all_points_R, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, img_size, flags=cv.CALIB_USE_INTRINSIC_GUESS)

    # 展示结果
    print('两个相机坐标系的旋转矩阵：\n', R)
    print('两个相机坐标系的平移向量：\n', T)
