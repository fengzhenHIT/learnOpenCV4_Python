# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 创建两个黑白图像
    img1 = np.zeros((200, 200), dtype='uint8')
    img2 = np.zeros((200, 200), dtype='uint8')
    img1[50:150, 50:150] = 255
    img2[100:200, 100:200] = 255
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()

    # 进行逻辑运算
    Not = cv.bitwise_not(img1)
    And = cv.bitwise_and(img1, img2)
    Or = cv.bitwise_or(img1, img2)
    Xor = cv.bitwise_xor(img1, img2)
    img_Not = cv.bitwise_not(img)

    # 展示结果
    cv.imshow('img1', img1)
    cv.imshow('img2', img2)
    cv.imshow('Not', Not)
    cv.imshow('And', And)
    cv.imshow('Or', Or)
    cv.imshow('Xor', Xor)
    cv.imshow('Origin', img)
    cv.imshow('Img_Not', img_Not)
    cv.waitKey(0)
    cv.destroyAllWindows()
