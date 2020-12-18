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
    orb = cv.ORB_create(1000, 1.2, 8, 31, 0, 2, cv.ORB_HARRIS_SCORE, 31, 20)

    # 分别计算image1，image2的ORB特征点和描述子
    kps1, des1 = orb.detectAndCompute(image1, None, None)
    kps2, des2 = orb.detectAndCompute(image2, None, None)

    # 特征点匹配
    # 创建BFMatcher对象
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 输出匹配结果中最大值和最小值
    matches_list = []
    for match in matches:
        matches_list.append(match.distance)
    max_dist = max(matches_list)
    min_dist = min(matches_list)

    # 设定阈值，筛选出合适的匹配点对
    good_matches = []
    for match in matches:
        if match.distance <= max(2.0 * min_dist, 20.0):
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
