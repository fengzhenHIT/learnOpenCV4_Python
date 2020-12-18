# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    # 读取数据及标签
    train_data = cv.imread('./results/train_data.png', cv.COLOR_BGR2GRAY).astype('float32')
    train_labels = cv.imread('./results/train_label.png', cv.COLOR_BGR2GRAY).astype('int32')

    # 训练随机树
    dt = cv.ml.DTrees_create()
    # 相关参数设置（可以缺省，但是会影响准确率，读者可以自行取消注释观察结果）
    dt.setMaxDepth(8)                               # 树的最大深度
    dt.setCVFolds(0)                                # 交叉验证次数
    # dt.setUseSurrogates(False)                      # 是否建立替代分裂点
    # dt.setMinSampleCount(2)                          # 节点最小样本数量
    # dt.setUse1SERule(False)                         # 是否严格剪枝
    # dt.setTruncatePrunedTree(False)                 # 分支是否完全移除
    dt.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    retval, results = dt.predict(train_data)

    # 计算准确率
    matches = results == train_labels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / results.size
    print('DTrees模型分类的准确率为：{}%'.format(accuracy))

    # 测试模型对数字的识别
    test_img1 = cv.imread('./images/handWrite01.png', cv.IMREAD_GRAYSCALE)
    test_img2 = cv.imread('./images/handWrite02.png', cv.IMREAD_GRAYSCALE)
    # 判断是否成功读取图像
    if test_img1 is None or test_img2 is None:
        print('Failed to read handWrite01.png or handWrite02.png.')
        sys.exit()
    cv.imshow('img1', test_img1)
    cv.imshow('img2', test_img2)

    # 缩放到指定尺寸
    img1 = cv.resize(test_img1, (20, 20)).reshape((1, 400))
    img2 = cv.resize(test_img2, (20, 20)).reshape((1, 400))
    x = np.concatenate((img1, img2), axis=0)
    test_data = x.astype(np.float32)

    # 进行数字识别
    ret, result = dt.predict(test_data)

    # 展示结果
    for i in range(len(result)):
        print('第{}张图像的真实结果为：{}，预测结果为：{}'.format(i + 1, i + 1, int(result[i])))

    cv.waitKey()
    cv.destroyAllWindows()
