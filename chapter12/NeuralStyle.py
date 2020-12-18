# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    image = cv.imread('./images/lena.jpg')
    # 判断是否成功读取图像
    if image is None:
        print('Failed to read lena.jpg.')
        sys.exit()
    cv.imshow('Origin', image)
    # 计算图像均值
    image_mean = np.mean(image)
    # 计算图像尺寸
    h, w = image.shape[:-1]

    # 设置需要进行迁移的图像风格
    styles = ['the_wave.t7', 'mosaic.t7', 'feathers.t7', 'candy.t7', 'udnie.t7']

    for i in range(len(styles)):
        # 加载模型
        net = cv.dnn.readNet('./data/styles/{}'.format(styles[i]))

        # 调整图像尺寸
        blob = cv.dnn.blobFromImage(image, 1.0, size=(512, 512), mean=image_mean, swapRB=False, crop=False)
        # 计算网络对图像的处理结果
        net.setInput(blob)
        prob = net.forward()

        # 解析输出
        prob = prob.reshape(3, prob.shape[2], prob.shape[3])
        # 恢复图像减掉的均值
        prob += image_mean
        # 对图像进行归一化
        prob /= 255.0
        prob = prob.transpose(1, 2, 0)
        prob = np.clip(prob, 0.0, 1.0)
        cv.normalize(prob, prob, 0, 255, cv.NORM_MINMAX)

        # 调整到最终需要显示的图像尺寸
        result = np.uint8(cv.resize(prob, (w, h)))
        cv.imshow('{}'.format(styles[i]), result)

    cv.waitKey(0)
    cv.destroyAllWindows()
