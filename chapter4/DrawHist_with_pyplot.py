# -*- coding:utf-8 -*-
import cv2 as cv
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':
    # 以灰度模式读取图像
    img = cv.imread('./images/flower.jpg', 0)
    # 判断图片是否读取成功
    if img is None:
        print('Failed to read flower.jpg.')
        sys.exit()

    # 绘制直方图并展示
    _, _, _ = plt.hist(x=img.ravel(), bins=256, range=[0, 256])
    cv.imshow('image', img)
    plt.show()
    cv.waitKey(0)
    cv.destroyAllWindows()
