# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
import sys


if __name__ == '__main__':
    # 构建一个HSV格式颜色地图，然后将其转换为BGR格式
    hsv_map = np.zeros((180, 256, 3), dtype=np.uint8)
    h, s = np.indices(hsv_map.shape[:2])
    hsv_map[:, :, 0] = h
    hsv_map[:, :, 1] = s
    hsv_map[:, :, 2] = 255
    hsv_map = cv.cvtColor(hsv_map, cv.COLOR_HSV2BGR)

    # 读取图像road.jpg
    image = cv.imread('./images/road.jpg')
    # 判断是否读取成功
    if image is None:
        print("Failed to read road.jpg.")
        sys.exit()
    # 将图片由BGR格式转换成HSV格式
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # 计算2D直方图
    image_hist = cv.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    print('2D直方图计算结果：\n{}'.format(image_hist))
    image_hist = np.clip(image_hist * 0.05, 0, 1)
    result = hsv_map * image_hist[:, :, np.newaxis] / 255.0

    # 展示结果
    cv.imshow('Origin Image', image)
    cv.imshow('Hsv Map', hsv_map)
    cv.imshow('2D Hist', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
