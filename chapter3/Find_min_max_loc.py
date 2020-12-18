# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 新建矩阵array
    array = np.array([1, 2, 3, 4, 5, 10, 6, 7, 8, 9, 10, 0])
    # 将array调整为3*4的单通道图像
    img1 = array.reshape((3, 4))
    minval_1, maxval_1, minloc_1, maxloc_1 = cv.minMaxLoc(img1)
    print('图像img1中最小值为：{}, 其位置为：{}' .format(minval_1, minloc_1))
    print('图像img1中最大值为：{}, 其位置为：{}' .format(maxval_1, maxloc_1))

    # 先将array调整为为3*2*2的多通道图像
    img2 = array.reshape((3, 2, 2))
    # 再利用-1的方法调整尺寸
    img2_re = img2.reshape((1, -1))
    minval_2, maxval_2, minloc_2, maxloc_2 = cv.minMaxLoc(img2_re)
    print('图像img2中最小值为：{}, 其位置为：{}'.format(minval_2, minloc_2))
    print('图像img2中最大值为：{}, 其位置为：{}'.format(maxval_2, maxloc_2))
