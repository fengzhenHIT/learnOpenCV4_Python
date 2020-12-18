# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/ColorSaltPepperImage.jpg', cv.IMREAD_ANYCOLOR)
    gray = cv.imread('./images/GraySaltPepperImage.jpg', cv.IMREAD_ANYCOLOR)
    if img is None or gray is None:
        print('Failed to read ColorSaltPepperImage.jpg or ColorSaltPepperImage.jpg.')
        sys.exit()

    # 分别对含有椒盐噪声的彩色和灰度图像进行中值滤波，后面的数字代表滤波器尺寸
    img_3 = cv.medianBlur(img, 3)
    gray_3 = cv.medianBlur(gray, 3)
    # 加载滤波器尺寸，图像会变模糊
    img_9 = cv.medianBlur(img, 9)
    gray_9 = cv.medianBlur(gray, 9)

    # 展示结果
    cv.imshow('Origin img', img)
    cv.imshow('img 3*3', img_3)
    cv.imshow('img 9*9', img_9)
    cv.imshow('Origin gray', gray)
    cv.imshow('gray 3*3', gray_3)
    cv.imshow('gray 9*9', gray_9)
    cv.waitKey(0)
    cv.destroyAllWindows()
