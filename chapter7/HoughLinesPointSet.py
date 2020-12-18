# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 生成float32类型的20 * 2 矩阵表示2D点集
    points = np.array([[[0.0, 369.0], [10.0, 364.0], [20.0, 358.0], [30.0, 352.0], [40.0, 346.0],
                       [50.0, 341.0], [60.0, 335.0], [70.0, 329.0], [80.0, 323.0], [90.0, 318.0],
                       [100.0, 312.0], [110, 306.0], [120.0, 300.0], [130.0, 295.0], [140.0, 289.0],
                       [150.0, 284.0], [160.0, 277.0], [170.0, 271.0], [180.0, 266.0], [190.0, 260.0]]],
                      dtype='float32')

    # 设置参数
    min_rho = 0.0                        # 最小长度
    max_rho = 360.0                      # 最大长度
    rho_step = 1                         # 离散化单位距离长度
    min_theta = 0.0                      # 最小角度
    max_theta = np.pi / 2.0              # 最大角度
    theta_step = np.pi / 180.0           # 离散化单位角度弧度

    # 进行检测
    lines = cv.HoughLinesPointSet(points, 20, 1, min_rho, max_rho, rho_step, min_theta, max_theta, theta_step)
    for item in lines:
        print('votes: {}, rho: {}, theta: {}'.format(item[0][0], item[0][1], item[0][2])))
