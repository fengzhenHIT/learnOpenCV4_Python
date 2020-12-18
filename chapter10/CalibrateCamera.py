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
    obj_points = np.reshape(obj_points, (54, 1, 3))

    # 计算棋盘格内角点的三维坐标及其在图像中的二维坐标
    all_obj_points = []
    all_points = []
    for i in range(1, 5):
        # 读取图像
        image = cv.imread('./images/left0{}.jpg'.format(i))
        if image is None:
            print('Failed to read left0{}.jpg.'.format(i))
            sys.exit()

        # 获取图像尺寸
        h, w = image.shape[:2]
        # 计算三维坐标
        all_obj_points.append(obj_points)
        # 计算二维坐标
        all_points.append(compute_points(image))

    # 计算相机内参矩阵和畸变系数
    _, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(all_obj_points, all_points, (w, h), None, None)
    print('内参矩阵为：\n{}'.format(cameraMatrix))
    print('畸变系数为：\n{}'.format(distCoeffs))

    # 此结果在后面ProjectPoints.py函数中有使用到，此处先进行计算
    print('旋转向量为：\n{}'.format(rvecs))
    print('平移向量为：\n{}'.format(tvecs))
