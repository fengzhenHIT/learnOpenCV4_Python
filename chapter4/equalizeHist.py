# -*- coding:utf-8 -*-
import cv2 as cv
from matplotlib import pyplot as plt
import sys


if __name__ == '__main__':
    # 读取图像
    image = cv.imread('./images/equalizeHist.jpg', 0)
    # 判断图片是否读取成功
    if image is None:
        print('Failed to read equalizeHist.jpg.')
        sys.exit()
    # 绘制原图直方图
    plt.hist(image.ravel(), 256, [0, 256])
    plt.title('Origin Image')
    plt.show()
    # 进行均衡化并绘制直方图
    image_result = cv.equalizeHist(image)
    plt.hist(image_result.ravel(), 256, [0, 256])
    plt.title('Equalized Image')
    plt.show()
    # 展示均衡化前后的图片
    cv.imshow('Origin Image', image)
    cv.imshow('Equalized Image', image_result)

    cv.waitKey(0)
    cv.destroyAllWindows()
