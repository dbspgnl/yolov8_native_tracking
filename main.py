import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import subprocess
import torch
import torchvision
import argparse
from typing import Dict, List, Set, Tuple
import math
from ast import literal_eval
from tqdm import tqdm
from multiprocessing import freeze_support
from collections import deque
import threading
import datetime
from utils.plots_ import plot_one_box2

car_names = ['car', 'truck', 'bus', 'vehicle'] # 차량 info 종류
colors_bgr = [[17, 133, 254],[255, 156, 100],[11, 189, 128],[0, 255, 255]] # 차량 info bgr 색상
car_colors = ['#ff911d', '#649af2', '#7eac06', '#fef63c'] # 차량 라벨 색상
COLORS = sv.ColorPalette.from_hex(car_colors) # car, truck, bus, vehicle

current = datetime.datetime.now()
Timer = None
before_frame = 0
now_frame = 0
work_frame = 60

def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]

def current_time() -> None: # 시간 측정 기능
    global current, Timer, before_frame, now_frame, work_frame
    work_frame = now_frame - before_frame # 실제 프레임 (작업 프레임) 설정
    before_frame = now_frame
    Timer = threading.Timer(1, current_time) # 1초마다 재호출
    Timer.start() 


class FFmpegProcessor:
    def __init__(self, input_path:str, output_path:str, pts:str, threads:str) -> None:
        self.cap = cv2.VideoCapture(input_path)
        self.target:str = output_path
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pts = pts
        self.threads = threads
        
    def getWidth(self): return self.width
    def getHeight(self): return self.height
    def getFPS(self): return self.fps
    
    def setPath(self):
        command = ['ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', "{}x{}".format(self.width, self.height),
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-filter:v', 'setpts={}*PTS'.format(self.pts), 
                '-threads', self.threads,
                '-r', '30',
                '-f', 'flv',
                self.target]
        return subprocess.Popen(command, stdin=subprocess.PIPE)


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str,
        zone_in_polygons: str  = None,
        count_line: str  = None,
        is_count_show: str  = False,
        is_show: bool = False,
        is_file: bool = False,
        pts: str = '10.0', # 초당 처리 프레임
        threads: str = '4',
    ) -> None:
        # 인자값 설정
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.is_show = is_show
        self.is_file = is_file
        self.is_count_show = is_count_show
        self.pts = pts
        self.threads = threads
        # YOLO 설정
        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        # 영상 기본 정보
        self.video_info = sv.VideoInfo.from_video_path(video_path=self.source_video_path)
        self.width = self.video_info.width
        self.height = self.video_info.height
        # 카운팅 라인 처리
        self.line_annotate = None 
        self.line_counters = []
        if count_line == "": count_line = "[]"    
        if not not literal_eval(count_line): # 배열이 있으면 처리
            self.line_annotate = sv.LineZoneAnnotator(color=sv.Color.from_hex('#fc0303'), thickness=1, text_thickness=1, text_scale=0.4)
            arrays = literal_eval(count_line)
            for array in arrays:
                if len(array) == 0: continue
                LINE_START = sv.Point(array[0][0], array[0][1])
                LINE_END = sv.Point(array[1][0], array[1][1])
                points = sv.LineZone(start=LINE_START, end=LINE_END)
                self.line_counters.append(points)
        # 로직 분류    
        if is_file: # 파일 로직
            self.target_fps = round(self.video_info.fps)
        else: # 스트림 로직
            ffmpeg = FFmpegProcessor(source_video_path, target_video_path, pts, threads)
            self.process = ffmpeg.setPath()
            self.width = ffmpeg.getWidth()
            self.height = ffmpeg.getHeight()
            self.target_fps = 8 #ffmpeg.getFPS() #검출 후 송출하는 영상 pfs 기준 
        # 영역 처리
        if zone_in_polygons == "": zone_in_polygons = "[]"    
        if not literal_eval(zone_in_polygons): # 전체 영역
            self.zone_display = False
            self.zones_in = initiate_polygon_zones(
                [np.array([[0,0],[self.width,0],[self.width, self.height],[0,self.height]])], (self.width, self.height), sv.Position.CENTER
            )
        else: # 지정 영역
            self.zone_display = True
            array = literal_eval(zone_in_polygons)
            temp_array = []
            for arr in array:
                temp_array.append(np.array(arr))
            self.zones_in = initiate_polygon_zones(
                temp_array, (self.width, self.height), sv.Position.CENTER
            )
        self.zone_info = dict() # 각 영역(zone)에 해당하는 정보
        # 라벨링
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS, thickness=0)
        self.label_annotator = sv.LabelAnnotator(color=COLORS, text_scale=0.35, text_padding=2, 
            color_lookup=sv.ColorLookup.CLASS, text_color=sv.Color.white()
        )
        # JsonData
        self.identity = dict()
        self.detected_identity = dict() # 확인된 identity
        self.frame_number = 0
        self.counting = [] # 패널 카운팅
        self.twins = dict() # 오버레이 관리

    # 비디오 처리
    def process_video(self) -> None: 
        if self.is_file: self.file_process() # file로 로직 처리
        else: self.stream_process() # rtmp stream으로 처리
    
    
    def file_process(self) -> None:
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                result = self.model(frame, verbose=False, device=0, imgsz=self.video_info.width)[0]
                
                # detection 공통 작업: confidence & nms
                detections = self.common_process(result)

                # Annotation 처리
                annotated_frame = self.annotate_frame(frame, detections)

                sink.write_frame(annotated_frame)
                if self.is_show:
                    cv2.imshow("OpenCV View", annotated_frame)
                if (cv2.waitKey(1) == 27): # ESC > stop
                    break
            print(self.zone_info)


    def stream_process(self) -> None:    
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for result in self.model.track(source=self.source_video_path, show=False, stream=True, device=0, verbose=False, agnostic_nms=True, imgsz=1920):
                if result.boxes.id is None: # 검출이 안되면 스킵
                    continue
                frame = result.orig_img
                
                # detection 공통 작업: confidence & nms
                detections = self.common_process(result)

                # Tracker id
                if result.boxes is None or result.boxes.id is None:
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                    
                # Annotation 처리
                annotated_frame = self.annotate_frame(frame, detections)
                # Output 처리
                numpy_array = np.array(annotated_frame)
                
                try: 
                    if numpy_array is not None:
                        self.process.stdin.write(numpy_array.tobytes()) # Output FFmepg
                        if self.is_show:
                            cv2.imshow("OpenCV View", numpy_array)
                except Exception as e: # 오류 발생 시 subprocess 종료
                    print(e) 
                    self.process.terminate()
                    break
                
                if (cv2.waitKey(1) == 27): # ESC > stop
                    break
                
                
    def common_process(self, track) -> sv.Detections:
        global now_frame
        self.frame_number += 1 # 현재 프레임 카운팅
        now_frame = self.frame_number
        detections = sv.Detections.from_ultralytics(track)
        detections = detections[detections.confidence > 0.25] 
        detections = detections.with_nms(threshold=0.3) 
        detections = self.detect_in_area(detections) # 영역만 디텍팅
        detections = self.tracker.update_with_detections(detections=detections) # 새 번호 발급
        
        # 기록 데이터 세팅
        self.set_identity(detections)
        
        # 기록 데이터 삭제 정리 
        for k,v in self.identity.copy().items():
            if k not in self.identity:
                continue
            if self.identity[k]["frame"] + 2 < self.frame_number: # 3프레임보다 더 크면 제거
                self.identity.pop(k)
            for zone in self.zones_in: # 감지 영역에 걸치면 제거
                for line in zone.polygon:
                    already_del = False
                    if k not in self.identity:
                        already_del = True
                    if not already_del and (k in self.detected_identity) and self.is_line_over(line, self.identity[k]["center"]):
                        self.detected_identity.pop(k)
                        self.identity.pop(k)
                        self.delete_related_id(k)
        
        # 차량 카운팅
        self.counting = self.set_counting()
        
        # 한 번이라도 감지된 tracker_id 리스트
        for tracker_id in detections.tracker_id:
            if tracker_id not in self.detected_identity: # 신규 데이터 여부
                self.detected_identity[tracker_id] = self.frame_number
                if tracker_id not in self.identity:
                    continue # key 있는 경우에만
                for key, val in self.identity.copy().items():
                    if tracker_id != key:
                        if self.check_overlap(self.identity[tracker_id]["position"], val["position"]) > 0.5: # 오버레이
                            self.twins[tracker_id] = key # 오버레이 관리 (오버레이 대상끼리 id 동기화 작업)
                            target = self.identity[key] # 해당 id는 추적(target)로 관리
                            self.identity[tracker_id]["id"] = target["id"] 
                            self.identity[tracker_id]["class"] = target["class"] # car type number
                            self.identity[tracker_id]["class_type"] = target["class_type"] # car_name
        return detections
    
    
    def check_overlap(self, xyxy1: list, xyxy2: list) -> float: # 두 좌표의 영역 차이
        area = self.compute_intersect_area(xyxy1, xyxy2) # 신규 tracker_id의 좌표
        x, y = xyxy1[2] - xyxy1[0], xyxy1[3] - xyxy1[1]
        return area/(x*y)
        

    def compute_intersect_area(self, rect1, rect2) -> float:
        x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
        x3, y3, x4, y4 = rect2[0], rect2[1], rect2[2], rect2[3]
        # 우좌상하 벗어난 경우
        if x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4: return 0

        left_up_x, left_up_y = max(x1, x3), max(y1, y3)
        right_down_x, right_down_y = min(x2, x4), min(y2, y4)

        width = right_down_x - left_up_x
        height =  right_down_y - left_up_y
    
        return width * height

        
    def detect_in_area(self, detections: sv.Detections) -> sv.Detections:
        detections_in_zones = [] # Detect Area 감지 영역 처리
        for i, zone_in in enumerate(self.zones_in):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)        
        detections = sv.Detections.merge(detections_in_zones)
        return detections
    

    def is_line_over(self, line, xy) -> bool: # 라인에 걸치는 여부
        x, y = int(xy[0]), int(xy[1]) 
        is_x = True if line[0]-2 <= x <=line[0]+2 else False # 선 기준 +-2 두께에 닿으면
        is_y = True if line[1]-2 <= y <=line[1]+2 else False # 세로도 마찬가지
        is_min_x = True if 0 < x < 20 else False # 가로 해상도보다 20 작은 범위부터
        is_max_x = True if self.width-20 < x < self.width else False
        is_min_y = True if 0 < y < 20 else False # 세로 해상도보다 20 작은 범위부터
        is_max_y = True if self.height-20 < y < self.height else False
        return is_x or is_y or is_min_x or is_max_x or is_min_y or is_max_y
    

    def delete_related_id(self, id) -> None:
        for key, val in self.identity.copy().items(): # 해당 id로 되어있는 데이터 제거
            if val['id'] == id:
                self.identity.pop(key)

    # 화면 표기
    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        # 영역 테두리 처리
        if self.zone_display: # 영역 표기 처리일 때만
            for i, zone_in in enumerate(self.zones_in):
                annotated_frame = sv.draw_polygon(
                    annotated_frame, zone_in.polygon, sv.Color.from_hex('#ffffff')
                )
                sv.PolygonZoneAnnotator(
                    zone=zone_in,
                    color=sv.Color.from_hex('#ffffff'),
                    thickness=0.5,
                    text_thickness=1,
                    text_scale=1
                )

        # 라인 카운팅
        if len(self.line_counters) != 0:
            for i, line_counter in enumerate(self.line_counters):
                line_counter.trigger(detections=detections)
                for tracker_id in detections.tracker_id:

                    if tracker_id in line_counter.tracker_state and line_counter.tracker_state[tracker_id] == True:
                        detections_copy = detections.tracker_id.copy().tolist()
                        index = detections_copy.index(tracker_id)
                        class_id = detections.class_id[index]
                        
                        if i not in self.zone_info: self.zone_info[i] = {"class_count": {}, "in_count": 0, "out_count": 0, "total_count":0 }
                        temp_previous_count = self.zone_info[i]["total_count"]
                        self.zone_info[i]["in_count"] = line_counter.in_count
                        self.zone_info[i]["out_count"] = line_counter.out_count
                        self.zone_info[i]["total_count"] = (line_counter.in_count + line_counter.out_count)
                        
                        # 카운팅이 존재할 때 클래스 별로 카운팅 처리
                        if class_id not in self.zone_info[i]["class_count"]: self.zone_info[i]["class_count"][class_id] = 0
                        if temp_previous_count != self.zone_info[i]["total_count"]:
                            self.zone_info[i]["class_count"][class_id] +=1
                        
                if self.is_count_show: # 카운팅 라벨 표시
                    self.line_annotate.annotate(frame=annotated_frame, line_counter=line_counter)
        
        # 오버레이 검사
        cant = []
        for key, val in self.identity.items():
            if key in self.twins: # 오버레이 선별 처리 (겹치는 데이터 중에 선택)
                for new, target in self.twins.items():
                    if new not in self.identity or target not in self.identity:
                        continue
                    new_area1 = self.check_overlap(self.identity[new]["position"], self.identity[target]["position"])
                    new_area2 = self.check_overlap(self.identity[target]["position"], self.identity[target]["position"])
                    target_area1 = self.check_overlap(self.identity[target]["position"], self.identity[new]["position"])
                    target_area2 = self.check_overlap(self.identity[new]["position"], self.identity[new]["position"])
                    chk1 = (new_area1 + new_area2)/2
                    chk2 = (target_area1 + target_area2)/2
                    self.identity[new]["overlay"].append(chk1)
                    self.identity[target]["overlay"].append(chk2)
                    if chk1 > chk2: cant.append(target)
                    else: cant.append(new)
                    
        # 차량 라벨
        for key, val in self.identity.items():
            if key not in cant:
                plot_one_box2(
                    self.identity[key]["position"], 
                    annotated_frame, 
                    self.identity[key]["class_type"], 
                    label=f"[{self.identity[key]['id']}] {self.identity[key]['speed']}km", 
                    color=colors_bgr[self.identity[key]["class"]], 
                    line_thickness=1
                )
                
        # 차량 패널 표시
        self.set_info_panel(annotated_frame)

        return annotated_frame
    
    # 검출 정보로 JSON 데이터 수집
    def set_identity(self, detections: sv.Detections) -> None:
        global current, before_frame, work_frame
        none_detected_tracker_ids = list(set(self.detected_identity)-set(detections.tracker_id))
        # 비검출 데이터 처리
        for tracker_id in none_detected_tracker_ids:
            if tracker_id not in self.identity:
                continue # key 있는 경우에만
            xyxy = self.predict_xyxy(
                xyxys=self.identity[tracker_id]['position_array'], 
                gaps=self.identity[tracker_id]['position_gap'],
                direct=self.identity[tracker_id]['direct']
            )
            self.set_position(tracker_id, xyxy)
        
        # 검출 데이터 처리
        for i in range(len(detections.xyxy)):
            xyxy = detections.xyxy[i]
            tracker_id = detections.tracker_id[i]
            center = (round((xyxy[0]+xyxy[2])/2, 2), round((xyxy[1]+xyxy[3])/2,2))
            if tracker_id not in self.identity:
                self.identity[tracker_id] = {
                    "id": tracker_id, # 비추적
                    "position": xyxy,
                    "position_array": deque([xyxy], maxlen=20),
                    "position_gap_array": deque(maxlen=20),
                    "position_gap": [],
                    "overlay": deque(maxlen=3),
                    "direct": (0,0),
                    "frame": self.frame_number,
                    "class": detections.class_id[i], # 비추적
                    "class_type": car_names[detections.class_id[i]], # 비추적
                    "center": center,
                    "center_array": deque([center], maxlen=20),
                    "speed": 0,
                    "start_time": current, # 비추적
                    "now_time": current,
                }
            else:
                self.identity[tracker_id]['frame'] = self.frame_number
                self.set_position(tracker_id, xyxy)
                if self.identity[tracker_id]['speed'] == 0 and len(self.identity[tracker_id]['center_array']) > 2: # 속도가 0이면 즉시 속도 계산
                    speed = self.estimatespeed(self.identity[tracker_id]['center_array'][-1], self.identity[tracker_id]['center_array'][-2])
                    self.identity[tracker_id]['speed'] = speed
                elif self.frame_number > before_frame + work_frame : # 작업 프레임 단위 (영상 1초)마다 아래 갱신
                    self.identity[tracker_id]['now_time'] = current
                    if len(self.identity[tracker_id]['center_array']) > 2:
                        speed = self.estimatespeed(self.identity[tracker_id]['center_array'][-1], self.identity[tracker_id]['center_array'][-2])
                        self.identity[tracker_id]['speed'] = speed
    
    # 공통 좌표 구하기                    
    def set_position(self, tracker_id:int, xyxy:list) -> None:
        # xyxy 최대 범위는 -100 ~ 자기 해상도+100까지
        xyxy[0] = -100 if xyxy[0] < -100 else self.width + 100 if xyxy[0] > self.width + 100 else xyxy[0]
        xyxy[1] = -100 if xyxy[1] < -100 else self.height + 100 if xyxy[1] > self.height + 100 else xyxy[1]
        xyxy[2] = -100 if xyxy[2] < -100 else self.width + 100 if xyxy[2] > self.width + 100 else xyxy[2]
        xyxy[3] = -100 if xyxy[3] < -100 else self.height + 100 if xyxy[3] > self.height + 100 else xyxy[3]
        center = (round((xyxy[0]+xyxy[2])/2, 2), round((xyxy[1]+xyxy[3])/2,2))
        self.identity[tracker_id]['position'] = xyxy
        self.identity[tracker_id]['position_array'].append(xyxy)
        self.identity[tracker_id]['position_gap_array'].append(self.gap(self.identity[tracker_id]["position_array"][-1], self.identity[tracker_id]["position_array"][-2]))
        self.identity[tracker_id]['position_gap'] = self.avg_weight_gap(self.identity[tracker_id]['position_gap_array'])
        self.identity[tracker_id]['center'] = center
        self.identity[tracker_id]['center_array'].append(center)
        self.identity[tracker_id]['direct'] = self.direct(self.identity[tracker_id]["position_array"]) # 방향


    # 좌표값으로 속도 계산
    def estimatespeed(self, Location1:list, Location2:list) -> int:
        location_x = abs(Location2[0] - Location1[0])
        location_y = abs(Location2[1] - Location1[1])
        d_pixel = math.sqrt(math.pow(location_x, 2) + math.pow(location_y, 2))
        d_meters = (d_pixel/6.75) # 1080 해상도 거리 160m = 미터당 6.75픽셀
        speed = d_meters * self.target_fps * 3.6 # meter x fps x km
        return int(speed)
    
    # 이전 좌표 값으로 다음 좌표 값 예측
    def predict_xyxy(self, xyxys:list, gaps:list, direct:tuple) -> list[int]:
        if len(xyxys) < 2:
            return xyxys[-1]
        trend = "x" if abs(direct[0]) > abs(direct[1]) else "y" 
        if trend == "x": 
            return [xyxys[-1][0] + gaps[0], xyxys[-1][1], xyxys[-1][2] + gaps[2], xyxys[-1][3]]
        elif trend == "y":
            return [xyxys[-1][0], xyxys[-1][1] + gaps[1], xyxys[-1][2], xyxys[-1][3] + gaps[3]]
    
    # 마지막 좌표까지의 갭 차이
    def gap_xyxy(self, xyxys:list) -> list[int]:
        if len(xyxys) < 2: return [0,0,0,0]
        elif len(xyxys) == 2: return self.gap(xyxys[-1], xyxys[-2])
        else: return self.gap(xyxys[-1], self.avg_xyxy(xyxys))
        
    # 방향 튜플 구하기 xyxy > (-1,0) 좌측 이동 / (0,1) 하측 이동
    def direct(self, xyxys:list) -> tuple:
        if len(xyxys) <= 2:
            return (0,0)
        x1,y1,x2,y2 = self.gap(xyxys[-1], self.avg_xyxy(xyxys))
        return ((x1+x2)/2,(y1+y2)/2)
    
    # 두 개의 좌표 차이
    def gap(self, xyxy1:list, xyxy2:list) -> list[int]: #xyxy1이 더 최근
        x1 = xyxy1[0] - xyxy2[0]
        y1 = xyxy1[1] - xyxy2[1]
        x2 = xyxy1[2] - xyxy2[2]
        y2 = xyxy1[3] - xyxy2[3]
        return [x1,y1,x2,y2]
    
    # xyxyx의 누적 평균 좌표 구하기
    def avg_xyxy(self, xyxys2:list) -> list[int]:
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
    
    # gap 가중 평균
    def avg_weight_gap(self, xyxys:list) -> list[int]:
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
        
    # 차량 수 카운팅
    def set_counting(self) -> list[int]:
        count = [0 for i in range(len(car_names))]
        for k,v in self.identity.items():
            count[v["class"]] += 1
        return count
        
    # 패널 info 표시
    def set_info_panel(self, frame: np.ndarray) -> None:
        # 바탕
        white_color = (255, 255, 255)
        cv2.rectangle(frame, (self.width-200, 0), (self.width, 120), white_color, -1, cv2.LINE_AA) # 패널 크기 (200, 120)
        tl = 1 or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
        for i in range(len(car_names)):
            # 글자
            c1 = (self.width-180, 22 + (i * 28)) # 상단 22부터 시작
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(car_names[i], 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(frame, '{} [{}]'.format(car_names[i], self.counting[i]), (c1[0], c1[1] - 2), 0, tl / 3, [0,0,0], thickness=tf, lineType=cv2.LINE_AA)
            # 색상바
            cf1 = c1[0] + 100, c1[1]
            cf2 = cf1[0] + 50, c1[1] - t_size[1] - 3
            cv2.rectangle(frame, cf1, cf2, colors_bgr[i], -1, cv2.LINE_AA)
    
    
if __name__ == "__main__":
    freeze_support()
    current_time() # 시간 측정 시작
    parser = argparse.ArgumentParser(description="Traffic Flow Analysis with YOLOv8")
    parser.add_argument("--source_weights_path", required=True, type=str, help="Path to the source weights file")
    parser.add_argument("--source_video_path", required=True, type=str, help="Path to the source video file")
    parser.add_argument("--target_video_path", required=True, type=str, help="Path to the target video file (output)")
    parser.add_argument("--zone_in_polygons", default=None, type=str, help="Coordinate array for detection area")
    parser.add_argument("--count_line", default=None, type=str, help="Line zone coordinates for entry/exit counting processing")
    parser.add_argument("--count_show", default=False, type=str, help="Line zone Count Label Show")
    parser.add_argument("--show", default=False, type=str, help="OpenCV Show")
    parser.add_argument("--file", default=False, type=str, help="Make File")
    parser.add_argument("--pts", default="10.0", type=str, help="FFmpeg PTS set value")
    parser.add_argument("--threads", default="4", type=str, help="FFmpeg threads set value")
    args = parser.parse_args()
    count_show_bool_true = (args.count_show == 'true')
    show_bool_true = (args.show == 'true')
    file_bool_true = (args.file == 'true')
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        zone_in_polygons=args.zone_in_polygons,
        count_line=args.count_line,
        is_count_show=count_show_bool_true,
        is_show=show_bool_true,
        is_file=file_bool_true,
        pts=args.pts,
        threads=args.threads,
    )
    processor.process_video()
    Timer.cancel() # 타임쓰레드 반드시 종료