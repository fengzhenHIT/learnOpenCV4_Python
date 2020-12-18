# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 新建矩阵array
    array = np.array([1, 2, 3, 4, 5, 10, 6, 7, 8, 9, 10, 0])
    # 将array调整为3*4的单通道图像img1
    img1 = array.reshape((3, 4))
    # 将array调整为3*2*2的多通道图像img2
    img2 = array.reshape((3, 2, 2))

    # 分别计算图像img1和图像img2的平均值和标准差
    mean_img1 = cv.mean(img1)
    mean_img2 = cv.mean(img2)

    mean_std_dev_img1 = cv.meanStdDev(img1)
    mean_std_dev_img2 = cv.meanStdDev(img2)

    # 输出cv.mean()函数计算结果
    print('cv.mean()函数计算结果如下：')
    print('图像img1的均值为：{}'.format(mean_img1))
    print('图像img2的均值为：{}\n第一个通道的均值为：{}\n第二个通道的均值为：{}'
          .format(mean_img2, mean_img2[0], mean_img2[1]))
    print('*' * 30)
    # 输出cv.meanStdDev()函数计算结果
    print('cv.meanStdDev()函数计算结果如下：')
    print('图像img1的均值为：{}\n标准差为：{}'.format(mean_img1[0], float(mean_std_dev_img1[1])))
    print('图像img2的均值为：{}\n第一个通道的均值为：{}\n第二个通道的均值为：{}\n'
          '标准差为：{}\n第一个通道的标准差为：{}\n第二个通道的标准差为：{}\n'
          .format(mean_img2, mean_img2[0], mean_img2[1],
                  mean_std_dev_img2[1], float(mean_std_dev_img2[1][0]), float(mean_std_dev_img2[1][0])))
