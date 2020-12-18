# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
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

    # 创建BFMatcher对象
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    # 暴力匹配
    matches = bf.match(des1, des2)

    # 查找最小汉明距离
    matches_list = []
    for match in matches:
        matches_list.append(match.distance)
    min_dist = min(matches_list)

    # 设定阈值，筛选出合适汉明距离的匹配点对
    good_matches = []
    for match in matches:
        if match.distance <= max(2.0 * min_dist, 20.0):
            good_matches.append(match)

    # 使用RANSAC算法筛选匹配结果
    # 获取关键点坐标
    src_kps = np.float32([kps1[i.queryIdx].pt for i in good_matches]).reshape(-1, 1, 2)
    dst_kps = np.float32([kps2[i.trainIdx].pt for i in good_matches]).reshape(-1, 1, 2)

    # 使用RANSAC算法筛选
    M, mask = cv.findHomography(src_kps, dst_kps, method=cv.RANSAC, ransacReprojThreshold=5.0)

    # 保存筛选后的匹配点对
    good_ransac = []
    for i in range(len(mask)):
        if mask[i] == 1:
            good_ransac.append(good_matches[i])
    
    # 绘制筛选前后的匹配结果
    result = cv.drawMatches(image1, kps1, image2, kps2, good_ransac, None)

    # 展示结果
    cv.imshow('RANSAC Matches', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
