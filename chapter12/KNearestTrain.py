# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    image = cv.imread('./images/digits.png')
    # 判断是否成功读取图像
    if image is None:
        print('Failed to read digits.png.')
        sys.exit()

    # 转为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 读取图像中的数据并创建训练数据
    cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
    x = np.array(cells)
    # 创建训练数据
    train_data = x.reshape(-1, 400).astype(np.float32)
    # 创建训练标签
    k = np.arange(10)
    train_labels = np.repeat(k, 500)[:, np.newaxis]

    # 加载训练数据集
    knn = cv.ml.KNearest_create()
    # 每个类别拿出5个数据
    knn.setDefaultK(5)
    # 设置为分类训练
    knn.setIsClassifier(True)
    # 训练KNN
    a = knn.train(train_data, cv.ml.ROW_SAMPLE, train_labels)

    # 保存手写数据、标签和训练结果
    cv.imwrite('./results/train_data.png', train_data)
    cv.imwrite('./results/train_label.png', train_labels)
    knn.save('./results/knn_model.yml')
