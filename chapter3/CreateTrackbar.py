# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def call_backl_brightness(x):
    global value, img, img1
    value = cv.getTrackbarPos('brightness', 'Brighter')
    img1 = np.uint8(np.clip((value / 100 * img), 0, 255))


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/lena.jpg')
    img1 = img.copy()
    if img is None:
        print('Failed to read lena.jpg.')
        sys.exit()
    cv.namedWindow('Brighter')
    # 设置滑动条的初始值为100
    value = 100
    # 创建滑动条
    cv.createTrackbar('brightness', 'Brighter', value, 300, call_backl_brightness)

    while True:
        cv.imshow('Brighter', img1)
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
