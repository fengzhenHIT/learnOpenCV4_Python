# -*- coding:utf-8 -*-
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    # 读取数据及标签
    train_data = cv.imread('./results/train_data.png', cv.COLOR_BGR2GRAY).astype('float32')
    train_labels = cv.imread('./results/train_label.png', cv.COLOR_BGR2GRAY).astype('int32')

    # 训练随机树
    rt = cv.ml.RTrees_create()
    # 相关参数设置（可以缺省以提高运行速度，但是会影响准确率，读者可以自行注释观察结果）
    rt.setTermCriteria((cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.01))
    rt.setMaxDepth(10)                              # 树的最大深度
    rt.setMinSampleCount(10)                        # 设置最小样本数
    rt.setCVFolds(0)                                # 交叉验证次数
    rt.setRegressionAccuracy(0)                     # 回归算法精度
    rt.setUseSurrogates(False)                      # 是否使用代理
    rt.setMaxCategories(15)                         # 最大类别数
    rt.setCalculateVarImportance(True)              # 是否需要计算Var
    rt.setActiveVarCount(4)                         # 设置Var的数目
    rt.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
    retval, results = rt.predict(train_data)

    # 计算准确率
    matches = results == train_labels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / results.size
    print('RTrees模型分类的准备率为：{}%'.format(accuracy))

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
    ret, result = rt.predict(test_data)

    # 展示结果
    for i in range(len(result)):
        print('第{}张图像的真实结果为：{}，预测结果为：{}'.format(i + 1, i + 1, int(result[i])))

    cv.waitKey()
    cv.destroyAllWindows()
