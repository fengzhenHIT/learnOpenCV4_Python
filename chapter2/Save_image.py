# -*- coding:utf-8 -*-
import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 读取图像并判断是否读取成功
    img = cv.imread('./images/flower.jpg')
    if img is None:
        print('Failed to read flower.jpg.')
        sys.exit()
    else:
        # 添加alpha通道（cv.merge()函数将在第三章做具体讲解）
        zeros = np.ones(img.shape[:2], dtype=img.dtype) * 100
        result = cv.merge([img, zeros])
        print('原图的通道数为：{}'.format(img.shape[2]))
        print('处理后的通道数为：{}'.format(result.shape[2]))

        # 图像展示
        plt.imshow(result)
        plt.show()

        # 图像保存
        cv.imwrite('./results/flower_alpha.png', result)
