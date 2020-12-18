# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/flower.jpg')
    if img is None:
        print('Failed to read flower.jpg.')
        sys.exit()
    else:
        print('图像的形状为：{}\n元素数据类型为：{}\n图像通道数为：{}\n像素总数为：{}'
              .format(img.shape, img.dtype, img.ndim, img.size))
