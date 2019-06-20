import cv2
import numpy as np
import apply as ap
from nms import nms

def First_method(img_path, clf, model, position):#mser方法
    position = ap.exchange(position)
    img = cv2.GaussianBlur(cv2.imread(img_path), (5, 5), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img_gray)
    contours = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    contours = ap.Screen_Area(contours, img_gray.shape)#面积筛选
    contours = ap.Screen_ratio(contours)#孔的长宽比筛选
    rects, contours = ap.Surrounding_Rects(contours)#获取包围矩形
    rects, contours = ap.Remove_overlap(img, rects, contours, position)#去掉重叠矩形，并将服务器范围内的矩形去掉
    rects_left, rects_right = ap.Get_coverage(img, rects, position)#获取可能存在的字符区域
    rects_character = ap.Get_character(rects_left, rects_right, img, clf)#获取字符区域
    rects_character_1 = ap.screen_character(rects_character, img, model)#进一步筛选能够正确识别出来的字符
    if len(rects_character_1[0]) >= 2 or len(rects_character_1[1]) >= 2:
        k, b = ap.first_method(rects_character_1)
        scale, standard = ap.Get_scale(rects_character_1)
        prediction = ap.position_pre(k, b, scale, standard, position)
        return prediction
    elif len(rects_character_1[0]) >= 1 or len(rects_character_1[1]) >= 1:
        prediction = ap.second_method(rects_character_1, position)
        return prediction
    else:
        return [0]


