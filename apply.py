import cv2
import os
import numpy as np
import math
import detect
import itertools
from nms import nms
from scipy.optimize import leastsq


def exchange(position):
    if type(position[0]).__name__ != 'list':
        x_min = np.int0(position[0])
        y_min = np.int0(position[1])
        x_max = np.int0(position[0] + position[2])
        y_max = np.int0(position[1] + position[3])
        return [[x_min, y_min, x_max, y_max]]
    else:
        location = []
        for c in position:
            x_min = np.int0(c[0])
            y_min = np.int0(c[1])
            x_max = np.int0(c[0] + c[2])
            y_max = np.int0(c[1] + c[3])
            location.append([x_min, y_min, x_max, y_max])
        return location

def value_compare(value1, value2):#比较值，用于计算图像边缘
    if value1 > value2:
        value = value2
    elif value1 >= 0:
        value = value1
    else:
        value = 0
    return value

def larger(s1, s2):#取维度较大的对象
    if len(s1) >= len(s2):
        return s1
    else:
        return s2

def error(p, x, y):#最小二乘法
    return func(p, x) - y

def func(p, x):#最小二乘法
    k, b = p
    return k * x + b

def Imshow(img):#显示图片
    cv2.namedWindow('IMG', cv2.WINDOW_NORMAL)
    cv2.imshow('IMG', img)
    cv2.waitKey(0)

def draw_contours(img,contours):#画轮廓的包围矩形
    for c in contours:
        x, y, w, h = np.int0(cv2.boundingRect(c))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    Imshow(img)

def draw_rect(img, rects):#画矩形
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    Imshow(img)

def local_threshold(img):#局部自适应二值化
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 20)
    return binary

def OTSU_threshold(img):#大津阈值法
    ret,binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return binary

def Screen_Area(contours, shape):#面积筛选
    result = []
    h, w = shape
    for c in contours:
        area = cv2.contourArea(c)
        rate = area / (w * h)
        if rate >= 2.46 * math.pow(10, -4) and rate <= 8.38 * math.pow(10, -4):
            result.append(c)
    return result

def Screen_ratio(contours):#长宽比筛选
    result = []
    for c in contours:
        boxTemp = cv2.minAreaRect(c)
        pt = cv2.boxPoints(boxTemp)
        pt = np.int0(pt)
        axisLongTemp = np.sqrt(pow(pt[1][0] - pt[0][0], 2) + pow(pt[1][1] - pt[0][1], 2))
        axisShortTemp = np.sqrt(pow(pt[2][0] - pt[1][0], 2) + pow(pt[2][1] - pt[1][1], 2))
        if axisShortTemp > axisLongTemp:
            LengthTemp = axisLongTemp
            axisLongTemp = axisShortTemp
            axisShortTemp = LengthTemp
        if abs(axisLongTemp / axisShortTemp - 1.2) <= 0.2:
            result.append(c)
    return result

def Surrounding_Rects(contours):#包围矩形
    result_1 = []
    result_2 = []
    for c in contours:
        rect = np.int0(cv2.boundingRect(c))
        result_1.append(rect)
        result_2.append(c)
    return result_1, result_2

def Remove_overlap(img, rects, contours, position):#去掉重叠框，并将服务器范围内的框去掉
    temp_1 = []
    for x, y, x1, y1 in position:
        temp = [np.int0(x + (x1 - x)*0.05), np.int0(y + (y1 - y) * 0.05), np.int0((x1 - x)*0.9), np.int0((y1 - y) * 0.9)]
        temp_1.append(temp)
        rects.append(temp)
        contours.append(temp)
    result_1, result_2 = nms(rects, contours, 0.05)
    #draw_rect(img, result_1)
    for c in result_1:
        if c in temp_1:
            result_1.remove(c)
    for c in result_2:
        if c in temp_1:
            result_2.remove(c)
    return [result_1, result_2]

def Get_coverage(img, rects, position):#获取可能存在的字符区域
    result_left = []
    result_right = []
    h1, w1, ret = img.shape
    x, y, x_1, y_1 = position[0]
    x_main = np.int0((x + x_1) / 2)
    for c in rects:
        x, y, w, h = c
        if x <= x_main:
            x1 = np.int0(x)
            y1 = value_compare(np.int0(y - 0.1 * h), h1)
            x2 = value_compare(np.int0(x1 - 1.12 * w), w1)
            y2 = value_compare(np.int0(y1 + 1.2 * h), h1)
            if x1 != x2 and y1 != y2:
                temp = x2
                x2 = x1
                x1 = temp
                result_left.append([c, [x1, y1, x2 - x1, y2 - y1]])
        else:
            x1 = value_compare(np.int0(x + 1 * w), w1)
            y1 = value_compare(np.int0(y - 0.1 * h), h1)
            x2 = value_compare(np.int0(x1 + 1.12 * w), w1)
            y2 = value_compare(np.int0(y1 + 1.2 * h), h1)
            if x1 != x2 and y1 != y2:
                result_right.append([c, [x1, y1, x2 - x1, y2 - y1]])
    return [result_left, result_right]

def Get_slope(rects):#计算斜率，截距
    reg = np.array(rects)[:, 0]
    reg = np.array(reg.tolist())
    X = reg[:, 0]
    Y = reg[:, 1]
    P = [1, 20]
    Para = leastsq(error, P, args=(X, Y))
    return Para[0]

def predict(clf, img):#AdaBoost对字符、非字符二分类
    img_c = cv2.resize(img, (28, 28))
    img_gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    features = img_gray.reshape(784).reshape(1, -1)
    classify = clf.predict(features)
    return classify[0]

def Get_effective(img, clf, shape):#筛选字符区域
    amount_1 = 0
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_thresh = local_threshold(img_gray)
    h, w = img_gray.shape
    for i in range(w):
        for j in range(h):
            if img_thresh[j, i] == 0:
                amount_1 += 1
    rate_1 = amount_1 / (w * h)
    w_main = shape[1]
    h, w, _ = img.shape
    if w / w_main >= 0.01488 and rate_1 >= 0.035 and predict(clf, img):
        return 1
    else:
        return 0

def Get_character(rects_left, rects_right, img, clf):#筛选字符区域
    left = []
    right = []
    for c in rects_left:
        x, y, w, h = c[1]
        img_test = img[y:(y+h), x:(x+w)]
        if Get_effective(img_test, clf, img.shape) == 1:
            left.append(c)
    for c in rects_right:
        x, y, w, h = c[1]
        img_test = img[y:(y+h), x:(x+w)]
        if Get_effective(img_test, clf, img.shape) == 1:
            right.append(c)
    return [left, right]

def screen_num(num_pre):#对字符识别结果进行筛选
    length, num = num_pre
    if length == 2 and (num >= 0 and num <= 50):
        return 1
    else:
        return 0

def screen_character(rects_character, img, model):#以识别的效果，进一步筛选字符
    result_left = []
    result_right = []
    left = rects_character[0]
    right = rects_character[1]
    for c in left:
        x, y, w, h = c[1]
        x1 = np.int0(x + w/2)
        y1 = np.int0(y + h/2)
        img_test = img[y:(y + h), x:(x + w)]
        num_pre = recognition(img_test, model)
        #print(num_pre[1])
        if screen_num(num_pre):
            result_left.append([[x1, y1], h, num_pre[1]])
    for c in right:
        x, y, w, h = c[1]
        x1 = np.int0(x + w/2)
        y1 = np.int0(y + h/2)
        img_test = img[y:(y + h), x:(x + w)]
        num_pre = recognition(img_test, model)
       #print(num_pre[1])
        if screen_num(num_pre):
            result_right.append([[x1, y1], h, num_pre[1]])
    return [result_left, result_right]

def first_method(rects_character_1):#获取斜率，截距
    left = rects_character_1[0]
    right = rects_character_1[1]
    if len(left) >= len(right):
        region = left
    else:
        region = right
    return Get_slope(region)

def recognition(img, model):#字符识别函数
    if os.path.exists('Test_img') == False:
        os.mkdir('Test_img')
    img_path = os.path.join('Test_img', 'img.png')
    cv2.imwrite(img_path, img)
    num = detect.infer(img_path, model)
    return num

def distance_count(point1, point2):#像素点距离计算
    distance = np.sqrt(pow((point1[0] - point2[0]), 2) + pow((point1[1] - point2[1]), 2))
    return distance

def Get_scale(rects_character_1):#获取换算比例
    scale = []
    left = rects_character_1[0]
    right = rects_character_1[1]
    if len(left) >= len(right):
        result = left
    else:
        result = right
    result_array = itertools.combinations(result, 2)
    for c in result_array:
        point1 = c[0]
        point2 = c[1]
        distance_img = distance_count(point1[0], point2[0])
        scale.append(abs(point1[2] - point2[2]) / distance_img)
    rand = np.random.rand(len(result))
    index = np.argsort(rand)[0]
    img_standard = result[index]

    return [np.mean(scale), img_standard]

def compute(k, b, scale, point, reg, point_main):#计算服务器实际位置
    if k != np.inf:
        y = point_main[1]
        x = (y - b)/k
    else:
        x = point[0]
        y = point_main[1]
    distance = distance_count((x, y), point)
    if y >= point[1]:
        result_position = round(reg - distance * scale)
    else:
        result_position = round(reg + distance * scale)
    return result_position

def position_pre(k, b, scale, standard, position):#计算服务器实际位置
    num = []
    point, _, reg = standard
    for x, y, x1, y1 in position:
        x_main = np.int0((x + x1) / 2)
        y_main = np.int0((y + y1) / 2)
        point_main = [x_main, y_main]
        num.append(compute(k, b, scale, point, reg, point_main))
    return num

def compute_1(element, point_main):
    point = element[0]
    height = element[1]
    reg = element[2]
    n = abs(point_main[1] - point[1]) / height
    num = n / 4.5
    if point_main[1] >= point[1]:
        result = reg - num
    else:
        result = reg + num
    return result

def position_pre_1(element, position):
    num = []
    for x, y, x1, y1 in position:
        x_main = np.int0((x + x1) / 2)
        y_main = np.int0((y + y1) / 2)
        point_main = [x_main, y_main]
        num.append(compute_1(element, point_main))
    return num

def second_method(rects_character_1, position):#第二种计算服务器位置方法
    result_pre = []
    result_pre_1 = 0
    left = rects_character_1[0]
    right = rects_character_1[1]
    result = left + right
    for c in result:
        result_pre.append(position_pre_1(c, position))#多层列表

    for c in result_pre:
        c = np.array(c)
        result_pre_1 = result_pre_1 + c
    result_pre_1 = result_pre_1 / (len(result_pre))

    for i in range(len(result_pre_1)):
        result_pre_1[i] = math.floor(result_pre_1[i])
    result_pre_2 = [int(i) for i in result_pre_1]

    return result_pre_2


