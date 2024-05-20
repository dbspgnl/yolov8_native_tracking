'''
좌표값을 구하는 함수 모음
'''
import math

import numpy as np


# 두 좌표값의 영역의 차이 
def check_overlap(xyxy1: list, xyxy2: list) -> float:
    area = compute_intersect_area(xyxy1, xyxy2) # 신규 tracker_id의 좌표
    x, y = xyxy1[2] - xyxy1[0], xyxy1[3] - xyxy1[1]
    return area/(x*y)
    

# 두 사각형의 범위 계산
def compute_intersect_area(rect1, rect2) -> float:
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    x3, y3, x4, y4 = rect2[0], rect2[1], rect2[2], rect2[3]
    # 우좌상하 벗어난 경우
    if x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4: return 0

    left_up_x, left_up_y = max(x1, x3), max(y1, y3)
    right_down_x, right_down_y = min(x2, x4), min(y2, y4)

    width = right_down_x - left_up_x
    height =  right_down_y - left_up_y

    return width * height


# 라인에 걸치는 여부
def is_line_over(line, xy, width, height) -> bool: 
    x, y = int(xy[0]), int(xy[1]) 
    is_x = True if line[0]-2 <= x <=line[0]+2 else False # 선 기준 +-2 두께에 닿으면
    is_y = True if line[1]-2 <= y <=line[1]+2 else False # 세로도 마찬가지
    is_min_x = True if 0 < x < 20 else False # 가로 해상도보다 20 작은 범위부터
    is_max_x = True if width-20 < x < width else False
    is_min_y = True if 0 < y < 20 else False # 세로 해상도보다 20 작은 범위부터
    is_max_y = True if height-20 < y < height else False
    return is_x or is_y or is_min_x or is_max_x or is_min_y or is_max_y


# 좌표값으로 속도 계산
def estimatespeed(Location1:list, Location2:list, target_fps) -> int:
    location_x = abs(Location2[0] - Location1[0])
    location_y = abs(Location2[1] - Location1[1])
    d_pixel = math.sqrt(math.pow(location_x, 2) + math.pow(location_y, 2))
    d_meters = (d_pixel/6.75) # 1080 해상도 거리 160m = 미터당 6.75픽셀
    speed = d_meters * target_fps * 3.6 # meter x fps x km
    return int(speed)


# 이전 좌표 값으로 다음 좌표 값 예측
def predict_xyxy(xyxys:list, gaps:list, direct:tuple) -> list[int]:
    if len(xyxys) < 2:
        return xyxys[-1]
    trend = "x" if abs(direct[0]) > abs(direct[1]) else "y" 
    if trend == "x": 
        return [xyxys[-1][0] + gaps[0], xyxys[-1][1], xyxys[-1][2] + gaps[2], xyxys[-1][3]]
    elif trend == "y":
        return [xyxys[-1][0], xyxys[-1][1] + gaps[1], xyxys[-1][2], xyxys[-1][3] + gaps[3]]


# 마지막 좌표까지의 갭 차이
def gap_xyxy(xyxys:list) -> list[int]:
    if len(xyxys) < 2: return [0,0,0,0]
    elif len(xyxys) == 2: return gap(xyxys[-1], xyxys[-2])
    else: return gap(xyxys[-1], avg_xyxy(xyxys))
    
        
# 두 개의 좌표 차이
def gap(xyxy1:list, xyxy2:list) -> list[int]: #xyxy1이 더 최근
    x1 = xyxy1[0] - xyxy2[0]
    y1 = xyxy1[1] - xyxy2[1]
    x2 = xyxy1[2] - xyxy2[2]
    y2 = xyxy1[3] - xyxy2[3]
    return [x1,y1,x2,y2]


# xyxyx의 누적 평균 좌표 구하기
def avg_xyxy(xyxys2:list) -> list[int]:
    xyxys = list(reversed(xyxys2))
    avg_array = [xyxys[0][0],xyxys[0][1],xyxys[0][2],xyxys[0][3]]
    for i in range(len(xyxys)):
        if i > 20: return avg_array
        xyxy = xyxys[i]
        avg_array[0] = (avg_array[0] + xyxy[0])/2
        avg_array[1] = (avg_array[1] + xyxy[1])/2
        avg_array[2] = (avg_array[2] + xyxy[2])/2
        avg_array[3] = (avg_array[3] + xyxy[3])/2
    return avg_array


# 방향 튜플 구하기 xyxy > (-1,0) 좌측 이동 / (0,1) 하측 이동
def direct(xyxys:list) -> tuple:
    if len(xyxys) <= 2:
        return (0,0)
    x1,y1,x2,y2 = gap(xyxys[-1], avg_xyxy(xyxys))
    return ((x1+x2)/2,(y1+y2)/2)


# gap 가중 평균
def avg_weight_gap(xyxys:list) -> list[int]:
    x1_arr, y1_arr, x2_arr, y2_arr = [], [], [], []
    for xyxy in xyxys:
        x1_arr.append(xyxy[0])
        y1_arr.append(xyxy[1])
        x2_arr.append(xyxy[2])
        y2_arr.append(xyxy[3])
    weights = [5]
    for i in range(len(x1_arr)-1):
        w = 5 - (i+1)
        if w < 1: w = 1
        weights.append(w)
    return [
        np.average(x1_arr, weights=(weights)), 
        np.average(y1_arr, weights=(weights)), 
        np.average(x2_arr, weights=(weights)), 
        np.average(y2_arr, weights=(weights))
    ]