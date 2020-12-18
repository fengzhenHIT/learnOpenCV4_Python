# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 填写模型和配置文件路径
    model = "./data/bvlc_googlenet.caffemodel"
    config = "./data/bvlc_googlenet.prototxt"

    # 加载模型
    net = cv.dnn.readNet(model, config)

    # 获取各层信息
    layer_names = net.getLayerNames()
    for name in layer_names:
        i = net.getLayerId(name)
        layer = net.getLayer(i)
        print('网络层数: {:<6} 网络层类型: {:<12} 网络层名称: {:<}'.format(i, layer.type, layer.name))
