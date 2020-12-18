#  -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


def add_noisy(image, n=10000):
    result = image.copy()
    w, h = image.shape[:2]
    for i in range(n):
        # 分别在宽和高的范围内生成一个随机值，模拟代表x, y坐标
        x = np.random.randint(1, w)
        y = np.random.randint(1, h)
        if np.random.randint(0, 2) == 0:
            # 生成白色噪声（盐噪声）
            result[x, y] = 0
        else:
            # 生成黑色噪声（椒噪声）
            result[x, y] = 255
    return result


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/dolphins.jpg')
    if img is None:
        print('Failed to read dolphins.jpg.')
        sys.exit()
    # 灰度图像添加椒盐噪声
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_image_noisy = add_noisy(gray_image, 10000)
    # 彩色图像添加椒盐噪声
    color_image_noisy = add_noisy(img, 10000)

    # 展示结果
    cv.imshow("Gray Image", gray_image)
    cv.imshow("Gray Image Noisy", gray_image_noisy)
    cv.imshow("Color Image", img)
    cv.imshow("Color Image Noisy", color_image_noisy)
    cv.waitKey(0)
    cv.destroyAllWindows()
