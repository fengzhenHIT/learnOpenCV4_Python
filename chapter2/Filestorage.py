# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


if __name__ == '__main__':
    # 创建FileStorage对象file，用于写入数据
    # 读者可以尝试将文件后缀名改为.yml或.yaml
    # file = cv.FileStorage('./data/MyFile.yml', cv.FileStorage_WRITE)
    # file = cv.FileStorage('./data/MyFile.yaml', cv.FileStorage_WRITE)
    file = cv.FileStorage('./data/MyFile.xml', cv.FileStorage_WRITE)

    # 写入数据
    file.write('name', '张三')
    file.write('age', 16)
    file.write('date', '2019-01-01')
    scores = np.array([[98, 99], [96, 97], [95, 98]])
    file.write('scores', scores)

    # 释放对象
    file.release()

    # 创建FileStorage对象file1，用于读取数据
    file1 = cv.FileStorage('./data/MyFile.xml', cv.FileStorage_READ)

    # 判断MyFile.xml文件是否成功打开
    if file1.isOpened():
        # 读取数据
        name1 = file1.getNode('name').string()
        age1 = file1.getNode('age').real()
        date1 = file1.getNode('date').string()
        scores1 = file1.getNode('scores').mat()

        # 展示读取结果
        print('姓名：{}'.format(name1))
        print('年龄：{}'.format(age1))
        print('记录日期：{}'.format(date1))
        print('成绩单：{}'.format(scores1))
    else:
        print('Can\'t open MyFile.xml.')

    # 释放对象
    file1.release()
