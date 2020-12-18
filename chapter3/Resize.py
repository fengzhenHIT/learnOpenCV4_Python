# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg', cv.IMREAD_GRAYSCALE)
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()

    # 将图像缩小
    small_img = cv.resize(img, (15, 15), fx=0, fy=0, interpolation=cv.INTER_AREA)
    # 最近邻插值
    big_img1 = cv.resize(small_img, (30, 30), fx=0, fy=0, interpolation=cv.INTER_NEAREST)
    # 双线性插值
    big_img2 = cv.resize(small_img, (30, 30), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
    # 双三次插值
    big_img3 = cv.resize(small_img, (30, 30), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    # 展示结果
    cv.namedWindow('small', cv.WINDOW_NORMAL)
    cv.imshow('small', small_img)
    cv.namedWindow('big_img1', cv.WINDOW_NORMAL)
    cv.imshow('big_img1', big_img1)
    cv.namedWindow('big_img2', cv.WINDOW_NORMAL)
    cv.imshow('big_img2', big_img2)
    cv.namedWindow('big_img3', cv.WINDOW_NORMAL)
    cv.imshow('big_img3', big_img3)
    cv.waitKey(0)
    cv.destroyAllWindows()
