# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    noobcv = cv.imread('./images/noobcv.jpg')
    if img is None or noobcv is None:
        print('Failed to read lena.jpg or noobcv.jpg.')
        sys.exit()
    mask = cv.resize(noobcv, (200, 200))
    # 深拷贝
    img1 = img.copy()
    # 浅拷贝
    img2 = img
    # 截取图像的ROI区域
    ROI = img[206: 406, 206: 406]
    # 深拷贝
    ROI_copy = ROI.copy()
    # 浅拷贝
    ROI1 = ROI
    img[206: 406, 206: 406] = mask
    # 展示结果
    cv.imshow('img + noobcv1', img1)
    cv.imshow('img + noobcv2', img2)
    cv.imshow('ROI copy1', ROI_copy)
    cv.imshow('ROI copy2', ROI1)

    # 在图像中绘制圆形
    img = cv.circle(img, (300, 300), 20, (0, 0, 255), -1)
    # 展示结果
    cv.imshow('img + circle1', img1)
    cv.imshow('img + circle2', img2)
    cv.imshow('ROI circle1', ROI_copy)
    cv.imshow('ROI circle2', ROI1)
    cv.waitKey(0)
    cv.destroyAllWindows()
