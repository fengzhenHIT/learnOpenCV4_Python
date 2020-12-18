# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    image = cv.imread('./images/people.jpg')
    # 判断是否成功读取图像
    if image is None:
        print('Failed to read people.jpg.')
        sys.exit()

    # 记录图像尺寸
    h, w, s = image.shape[::]

    # 定义一个用来填充分割后图像的色彩集合（注意此处的色彩数目不能少于图像分割的类别数）
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

    # 构建图像数据
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # 定义迭代算法终止条件
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置图像分割的类别
    num_clusters = 3
    # 图像分割
    ret, labels, centers = cv.kmeans(data, num_clusters, None, criteria, num_clusters, cv.KMEANS_RANDOM_CENTERS)
    # 为不同类别的图像区域根据定义的颜色集合进行填色
    for i in range(len(data)):
        data[i] = colors[int(labels[i])]

    # 展示结果
    result = data.reshape((h, w, s))
    cv.imshow('Origin', image)
    cv.imshow('Result', result)

    cv.waitKey(0)
    cv.destroyAllWindows()
