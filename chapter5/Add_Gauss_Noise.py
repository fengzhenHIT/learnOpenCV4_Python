#  -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def add_noise(image, mean=0, val=0.01):
    size = image.shape
    image = image / 255
    gauss = np.random.normal(mean, val ** 0.5, size)
    noise = image + gauss
    return gauss, noise


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/dolphins.jpg')
    if img is None:
        print('Failed to read dolphins.jpg.')
        sys.exit()
    # 灰度图像添加高斯噪声
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_gauss, gray_noisy_image = add_noise(gray_image)
    # 彩色图像添加高斯噪声
    color_gauss, color_noisy_image = add_noise(img)

    # 展示结果
    cv.imshow("Gray Image", gray_image)
    cv.imshow("Gray Gauss Image", gray_gauss)
    cv.imshow("Gray Noisy Image", gray_noisy_image)
    cv.imshow("Color Image", img)
    cv.imshow("Color Gauss Image", color_gauss)
    cv.imshow("Color Noisy Image", color_noisy_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
