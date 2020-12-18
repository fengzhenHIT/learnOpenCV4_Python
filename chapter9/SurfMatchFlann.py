# -*- coding:utf-8 -*-
import cv2 as cv
import sys


if __name__ == '__main__':
    # 读取图像
    image1 = cv.imread('./images/box.png')
    image2 = cv.imread('./images/box_in_scene.png')
    if image1 is None or image2 is None:
        print('Failed to read box.png or box_in_scene.png.')
        sys.exit()

    # 创建ORB对象
    surf = cv.xfeatures2d.SURF_create(500, 4, 3, True, False)

    # 分别计算image1，image2的ORB特征点和描述子
    kps1, des1 = surf.detectAndCompute(image1, None, None)
    kps2, des2 = surf.detectAndCompute(image2, None, None)

    # 判断描述子数据类型，若不符合则进行数据转换
    if des1.dtype is not 'float32':
        des1 = des1.astype('float32')
    if des2.dtype is not 'float32':
        des2 = des2.astype('float32')

    # 创建FlannBasedMatcher对象
    matcher = cv.FlannBasedMatcher()

    # 特征点匹配
    matches = matcher.match(des1, des2, None)

    # 寻找距离的最大值和最小值
    matches_list = []
    for match in matches:
        matches_list.append(match.distance)
    max_dist = max(matches_list)
    min_dist = min(matches_list)

    # 设定阈值，筛选出合适的匹配点对
    good_matches = []
    for match in matches:
        if match.distance < 0.4 * max_dist:
            good_matches.append(match)

    # 输出匹配成功的特征点数目
    print('匹配成功的特征点数目为：{}，筛选后的特征点数目为：{}'.format(len(matches), len(good_matches)))

    # 绘制筛选前后的匹配结果
    result1 = cv.drawMatches(image1, kps1, image2, kps2, matches, None)
    result2 = cv.drawMatches(image1, kps1, image2, kps2, good_matches, None)

    # 展示结果
    cv.imshow('Matches', result1)
    cv.imshow('Good Matches', result2)
    cv.waitKey(0)
    cv.destroyAllWindows()
