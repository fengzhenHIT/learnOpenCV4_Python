# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    # 读取数据
    points = cv.FileStorage('./data/point.yml', cv.FileStorage_READ)

    # 判断point.yml文件是否成功打开
    if points.isOpened():
        # 读取数据
        data = points.getNode('data').mat()
        labels = points.getNode('labls').mat()

        # 释放对象
        points.release()

        # 设置两种颜色以标注不同种类的坐标点
        colors = [(0, 255, 0), (0, 0, 255)]

        # 创建空白图像用于显示坐标点
        img = np.zeros((480, 640, 3), dtype='float32')
        img[::] = 255
        cv.imshow('Origin', img)
        for i in range(len(data)):
            x, y = data[i]
            cv.circle(img, (int(x), int(y)), 3, colors[int(labels[i])], -1)
        cv.imshow('Origin', img)

        # 建立模型
        svm = cv.ml.SVM_create()

        # 设置参数
        svm.setKernel(cv.ml.SVM_INTER)             # 内核的模型
        svm.setType(cv.ml.SVM_C_SVC)                # SVM的类型
        svm.setTermCriteria((cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.01))
        # svm.setGamma(5.383)
        # svm.setC(0.01)
        # svm.setDegree(3)

        # 训练模型
        svm.train(data, cv.ml.ROW_SAMPLE, labels)
        svm.save('./results/svm.dat')

        # 用模型对图像中全部像素点进行分类
        for i in range(0, 640, 2):
            for j in range(0, 480, 2):
                _, res = svm.predict(np.array([[i, j]], dtype='float32'))
                img[j, i] = colors[int(res)]
        # 展示分类预测结果
        cv.imshow('Result', img.astype('uint8'))
    else:
        print('Can\'t open point.yml.')

    cv.waitKey(0)
    cv.destroyAllWindows()
