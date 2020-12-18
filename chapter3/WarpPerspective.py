# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/noobcvqr.png')
    if img is None:
        print('Failed to read noobcvqr.png.')
        sys.exit()

    h, w = img.shape[:-1]
    size = (w, h)
    # 读取透视变换前四个角点坐标
    points_path = './data/noobcvqr_points.txt'
    with open(points_path, 'r') as f:
        src_points = np.array([tx.split(' ') for tx in f.read().split('\n')], dtype='float32')

    # 设置透视变换后四个角点坐标
    max_pt = np.max(src_points)
    dst_points = np.array([[0.0, 0.0], [max_pt, 0.0], [0.0, max_pt], [max_pt, max_pt]], dtype='float32')
    # 计算透视变换矩阵
    rotation = cv.getPerspectiveTransform(src_points, dst_points)
    # 透视变换投影
    img_warp = cv.warpPerspective(img, rotation, size)

    # 展示结果
    cv.imshow('Origin', img)
    cv.imshow('img_warp', img_warp)
    cv.waitKey(0)
    cv.destroyAllWindows()
