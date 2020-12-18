# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


if __name__ == '__main__':
    # 随机生成点集
    pts1 = np.random.randint(100, 200, (25, 2))
    pts2 = np.random.randint(300, 400, (25, 2))
    pts = np.vstack((pts1, pts2))

    # 初始化数据
    data = np.float32(pts)

    # 定义迭代算法终止条件
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 使用kmeans进行聚类
    ret, label, center = cv.kmeans(data, 2, None, criteria, 2, cv.KMEANS_RANDOM_CENTERS)

    # 输出结果
    for i in range(len(center)):
        print('第{}类的中心坐标：x={}  y={}'.format(i, int(center[i][0]), int(center[i][1])))

    # 获取不同标签的点
    A = data[label.ravel() == 0]
    B = data[label.ravel() == 1]

    # 绘制结果
    plt.scatter(A[:, 0], A[:, 1], s=10, c='r')
    plt.scatter(B[:, 0], B[:, 1], s=10, c='b')
    plt.scatter(center[:, 0], center[:, 1], s=20, c='g', marker='*')
    plt.scatter(center[:, 0], center[:, 1], c='', marker='o', edgecolors='g', s=5000)
    plt.xlabel('x'), plt.ylabel('y')
    plt.show()
