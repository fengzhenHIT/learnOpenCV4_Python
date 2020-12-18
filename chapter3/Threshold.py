# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 彩色图像二值化
    _, img_B = cv.threshold(img, 125, 255, cv.THRESH_BINARY)
    _, img_B_V = cv.threshold(img, 125, 255, cv.THRESH_BINARY_INV)
    cv.imshow('img_B', img_B)
    cv.imshow('img_B_V', img_B_V)
    # 灰度图像二值化
    _, gray_B = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    _, gray_B_V = cv.threshold(gray, 125, 255, cv.THRESH_BINARY_INV)
    cv.imshow('gray_B', gray_B)
    cv.imshow('gray_B_V', gray_B_V)
    # 灰度图像TOZERO变换
    _, gray_T = cv.threshold(gray, 125, 255, cv.THRESH_TOZERO)
    _, gray_T_V = cv.threshold(gray, 125, 255, cv.THRESH_TOZERO_INV)
    cv.imshow('gray_T', gray_T)
    cv.imshow('gray_T_V', gray_T_V)
    # 灰度图像TRUNC变换
    _, gray_TRUNC = cv.threshold(gray, 125, 255, cv.THRESH_TRUNC)
    cv.imshow('gray_TRUNC', gray_TRUNC)
    # 灰度图像大律法和三角形法二值化
    img1 = cv.imread('./images/threshold.png', cv.IMREAD_GRAYSCALE)
    _, img1_O = cv.threshold(img1, 100, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    _, img1_T = cv.threshold(img1, 125, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    cv.imshow('img1', img1)
    cv.imshow('img1_O', img1_O)
    cv.imshow('img1_T', img1_T)
    # 灰度图像自适应二值化
    adaptive_mean = cv.adaptiveThreshold(img1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 0)
    adaptive_gauss = cv.adaptiveThreshold(img1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 0)
    cv.imshow('adaptive_mean', adaptive_mean)
    cv.imshow('adaptive_gauss', adaptive_gauss)
    cv.waitKey(0)
    cv.destroyAllWindows()
