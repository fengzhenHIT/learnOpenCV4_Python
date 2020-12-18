# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import sys


if __name__ == '__main__':
    model = "./data/tensorflow_inception_graph.pb"
    label_path = "./data/imagenet_comp_graph_label_strings.txt"

    # 加载TensorFlow模型
    net = cv.dnn.readNet(model)

    # 获取模型对应标签
    with open(label_path, 'r') as f:
        label = f.readlines()

    image = cv.imread("./images/airplane.jpg")
    # 判断是否成功读取图像
    if image is None:
        print('Failed to read airplane.jpg.')
        sys.exit()

    # 调整图像尺寸
    blob = cv.dnn.blobFromImage(image, size=(224, 224), swapRB=True, crop=False)
    # 计算网络对图像的处理结果
    net.setInput(blob)
    prob = net.forward()

    # 获取最可能的分类输出及得分
    score = round(max(prob[0]) * 100, 4)
    class_name = label[np.argmax(prob[0])].split('\n')[0]
    string = '{}: {}'.format(class_name, score)

    # 展示结果
    cv.putText(image, string, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, 8)
    cv.imshow('Detect Result', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
