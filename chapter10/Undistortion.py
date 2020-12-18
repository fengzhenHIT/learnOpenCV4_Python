# -*- coding:utf-8 -*-
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

    # 依次校正图像
    for i in range(1, 5):
        # 读取图像
        img = cv.imread('./images/left0{}.jpg'.format(i))
        if img is None:
            print('Failed to read left0{}.jpg.'.format(i))
            sys.exit()

        # 获取图像尺寸
        h, w = img.shape[:2]

        # 校正图像（使用cv.initUndistortRectifyMap()函数和cv.remap()函数）
        map1, map2 = cv.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, None, (w, h), 5)
        result1 = cv.remap(img, map1, map2, cv.INTER_LINEAR)

        # 校正图像（使用cv.undistort()函数）
        result2 = cv.undistort(img, cameraMatrix, distCoeffs, newCameraMatrix=None)

        # 展示结果
        cv.imshow('Origin', img)
        cv.imshow('Result_left0{}.jpg(Mode 1)'.format(i), result1)
        cv.imshow('Result_left0{}.jpg(Mode 2)'.format(i), result2)
        k = cv.waitKey(0)

        # 设置点击enter键继续，其它键退出
        if k == 13:
            cv.destroyAllWindows()
        else:
            sys.exit()
