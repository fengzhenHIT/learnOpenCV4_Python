# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import datetime
import sys


if __name__ == '__main__':
    # 创建ndarray对象
    # 使用np.array()创建一个5*5，数据类型为float32的对象
    a = np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20],
                  [21, 22, 23, 24, 25]], dtype='float32')
    # 使用np.ones()创建一个5*5，数据类型为uint8的全1对象
    b = np.ones((5, 5), dtype='uint8')
    # 使用np.zeros()创建一个5*5，数据类型为float32的全0对象
    c = np.zeros((5, 5), dtype='float32')
    print('创建对象（np.array）：\n{}'.format(a))
    print('创建对象（np.ones）：\n{}'.format(b))
    print('创建对象（np.zeros）：\n{}'.format(c))

    # ndarray对象切片和索引
    image = cv.imread('./images/flower.jpg')
    # 判断图片是否读取成功
    if image is None:
        print('Failed to read flower.jpg.')
        sys.exit()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 读取图像位于（45,45）位置的像素点
    print('位于（45,45）位置的像素点为：{}'.format(gray[45, 45]))
    # 裁剪部分图像（灰度图像和RGB图像）
    res_gray = gray[40:280, 60:340]
    res_color1 = image[40:280, 60:340, :]
    res_color2 = image[100:220, 80:220, :]
    # 通道分离
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    # 展示裁剪和分离通道结果
    cv.imshow('Result crop gray', res_gray)
    cv.imshow('Result crop color1', res_color1)
    cv.imshow('Result crop color2', res_color2)
    cv.imshow('Result split b', b)
    cv.imshow('Result split g', g)
    cv.imshow('Result split r', r)

    # 生成随机数
    # 生成一个5*5，取值范围在0-100的数组
    values1 = np.random.randint(0, 100, (5, 5), dtype='uint8')
    # 生成一个2*3，元素服从平均值为0、标准差为1正态分布的数组
    values2 = np.random.randn(2, 3)
    print('生成随机数（np.random.randint）：\n{}'.format(values1))
    print('生成随机数（np.random.randn）：\n{}'.format(values2))

    cv.waitKey(0)
    cv.destroyAllWindows()
