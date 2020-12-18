# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    # 生成二值矩阵src
    src = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0],
                    [0, 255, 255, 255, 0, 255, 255, 255, 0, 0, 0, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0],
                    [0, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='uint8')

    # 生成3*3矩形结构元素
    kernel = cv.getStructuringElement(0, (3, 3))

    # 对二值矩阵分别进行开运算、闭运算、梯度运算、顶帽运算、黑帽运算以及击中击不中变换
    open_src = cv.morphologyEx(src, cv.MORPH_OPEN, kernel)
    close_src = cv.morphologyEx(src, cv.MORPH_CLOSE, kernel)
    gradient_src = cv.morphologyEx(src, cv.MORPH_GRADIENT, kernel)
    tophat_src = cv.morphologyEx(src, cv.MORPH_TOPHAT, kernel)
    blackhat_src = cv.morphologyEx(src, cv.MORPH_BLACKHAT, kernel)
    hitmiss_src = cv.morphologyEx(src, cv.MORPH_HITMISS, kernel)

    # 展示二值矩阵形态学操作结果
    cv.namedWindow('src', cv.WINDOW_NORMAL)
    cv.imshow('src', src)
    cv.namedWindow('Open src', cv.WINDOW_NORMAL)
    cv.imshow('Open src', open_src)
    cv.namedWindow('Close src', cv.WINDOW_NORMAL)
    cv.imshow('Close src', close_src)
    cv.namedWindow('Gradient src', cv.WINDOW_NORMAL)
    cv.imshow('Gradient src', gradient_src)
    cv.namedWindow('Tophat src', cv.WINDOW_NORMAL)
    cv.imshow('Tophat src', tophat_src)
    cv.namedWindow('Blackhat src', cv.WINDOW_NORMAL)
    cv.imshow('Blackhat src', blackhat_src)
    cv.namedWindow('Hitmiss src', cv.WINDOW_NORMAL)
    cv.imshow('Hitmiss src', hitmiss_src)
    cv.waitKey(0)

    # 读取图像keys.jpg并进行二值化
    keys = cv.imread('./images/keys.jpg', cv.IMREAD_GRAYSCALE)
    if keys is None:
        print('Failed to read keys.jpg.')
        sys.exit()
    cv.imshow('Origin', keys)
    keys = cv.threshold(keys, 130, 255, cv.THRESH_BINARY)[1]

    # 生成5*5矩形结构元素
    kernel_keys = cv.getStructuringElement(0, (5, 5))

    # 对图像分别进行开运算、闭运算、梯度运算、顶帽运算、黑帽运算以及击中击不中变换
    open_keys = cv.morphologyEx(keys, cv.MORPH_OPEN, kernel_keys)
    close_keys = cv.morphologyEx(keys, cv.MORPH_CLOSE, kernel_keys)
    gradient_keys = cv.morphologyEx(keys, cv.MORPH_GRADIENT, kernel_keys)
    tophat_keys = cv.morphologyEx(keys, cv.MORPH_TOPHAT, kernel_keys)
    blackhat_keys = cv.morphologyEx(keys, cv.MORPH_BLACKHAT, kernel_keys)
    hitmiss_keys = cv.morphologyEx(keys, cv.MORPH_HITMISS, kernel_keys)

    # 展示图像形态学操作结果
    cv.imshow('Two-valued keys', keys)
    cv.imshow('Open keys', open_keys)
    cv.imshow('Close keys', close_keys)
    cv.imshow('Gradient keys', gradient_keys)
    cv.imshow('Tophat keys', tophat_keys)
    cv.imshow('Blackhat keys', blackhat_keys)
    cv.imshow('Hitmiss keys', hitmiss_keys)

    cv.waitKey(0)
    cv.destroyAllWindows()
