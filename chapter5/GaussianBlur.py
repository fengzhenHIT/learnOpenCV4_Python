# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/Gray_dolphins.jpg', cv.IMREAD_ANYDEPTH)
    img_gauss = cv.imread('./images/GrayGaussImage.jpg', cv.IMREAD_ANYDEPTH)
    img_salt = cv.imread('./images/GraySaltPepperImage.jpg', cv.IMREAD_ANYDEPTH)
    if img is None or img_gauss is None or img_salt is None:
        print('Failed to read Gray_dolphins.jpg or GrayGaussImage.jpg or GraySaltPepperImage.jpg.')
        sys.exit()

    # 分别对上述图像进行高斯滤波，后面的数字代表滤波器尺寸
    result_5 = cv.GaussianBlur(img, (5, 5), 10, 20)
    result_9 = cv.GaussianBlur(img, (9, 9), 10, 20)
    result_5_gauss = cv.GaussianBlur(img_gauss, (5, 5), 10, 20)
    result_9_gauss = cv.GaussianBlur(img_gauss, (9, 9), 10, 20)
    result_5_salt = cv.GaussianBlur(img_salt, (5, 5), 10, 20)
    result_9_salt = cv.GaussianBlur(img_salt, (9, 9), 10, 20)

    # 展示结果
    cv.imshow('Origin img', img)
    cv.imshow('Result img 5*5', result_5)
    cv.imshow('Result img 9*9', result_9)
    cv.imshow('Origin img_gauss', img_gauss)
    cv.imshow('Result img_gauss 5*5', result_5_gauss)
    cv.imshow('Result img_gauss 9*9', result_9_gauss)
    cv.imshow('Origin img_salt', img_salt)
    cv.imshow('Result img_salt 5*5', result_5_salt)
    cv.imshow('Result img_salt 9*9', result_9_salt)
    cv.waitKey(0)
    cv.destroyAllWindows()
