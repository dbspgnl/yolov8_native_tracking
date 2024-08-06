import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
from typing import List, Tuple
from ast import literal_eval
from tqdm import tqdm
from multiprocessing import freeze_support, Process, Queue, Event
from collections import deque
from utils.plots_ import plot_one_box2
import utils.coord as coord
import utils.argument as arg
from utils.ffmpeg import FFmpegProcessor
import utils.timer as timer
from utils.xml import XML
from utils.extract import frame_extractor


import queue

car_names = ['car', 'truck', 'bus', 'vehicle'] # 차량 info 종류
colors_bgr = [[17, 133, 254],[255, 156, 100],[11, 189, 128],[0, 255, 255]] # 차량 info bgr 색상
car_colors = ['#ff911d', '#649af2', '#7eac06', '#fef63c'] # 차량 라벨 색상
COLORS = sv.ColorPalette.from_hex(car_colors) # car, truck, bus, vehicle


# 영역 처리 함수 : 감지 및 비감지 영역 지정 함수
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

def set_padding(width: int, height: int, padding: int) -> List[np.ndarray]:
    padding_polygons = []
    if padding > 0:
        w = width
        h = height
        p = padding
        padding_polygons = [
            np.array([[0, 0], [p, 0], [p, h], [0, h]]),  # Left padding
            np.array([[0, 0], [w, 0], [w, p], [0, p]]),  # Top padding
            np.array([[w, 0], [w, h], [w - p, h], [w - p, 0]]),  # Right padding
            np.array([[w, h], [0, h], [0, h - p], [w, h - p]])  # Bottom padding
        ]
    return padding_polygons
        
'''
__init__ : 입력 및 출력, YOLO모델, Tracker, video, 영역 등 초기화 / 카운팅 라인 및 검출 영역 설정
process_video : 비디오 처리 진입점 / 파일기반 처리 OR 스트림 기반 처리 (파일기반:file_process / 실시간기반:stream_process)
file_process: 비디오 파일을 프레임 단위로 처리, 1) Detection 공통작업(common_process) / 2) Annotation 처리된 영상 저장 / 3) xml file에 savedJson의 데이터를 저장 
common_process: 1. 현재 프레임 카운팅 / 2. 기록 데이터 세팅 (set_identity) / 3.검출정보를 savedJson에 JSON 형태로 저장
set_identity : Detections 데이터를 받아서 1. 비검출된 차량 예측 / 2. 검출 데이터 처리 / 3. 해당 데이터들을 "set_position"으로 self.identity에 전달
set_position : 1. 좌표값이 자기 해상도를 넘어가지 않도록 값을 변경 / 2. self.identity에 검출 데이터 전달

'''

class VideoProcessor:
    def __init__(self, arg: dict) -> None:
        # experimental part
        self.frame_queue = Queue(maxsize=10)
        # 인자값 설정
        self.source_video_path = arg["source_video_path"]
        self.target_video_path = arg["target_video_path"]
        self.is_show = arg["is_show"]
        self.is_file = arg["is_file"]
        self.is_count_show = arg["is_count_show"]
        self.pts = arg["pts"]
        self.threads = arg["threads"]
        self.keep_frame = int(arg["keep_frame"])
        self.tracker_yaml = arg["tracker_yaml"]
        # YOLO 설정
        self.model = YOLO(arg["source_weights_path"])
        self.tracker = sv.ByteTrack()
        # 영상 기본 정보
        self.video_info = sv.VideoInfo.from_video_path(video_path=self.source_video_path)
        self.width = self.video_info.width
        self.height = self.video_info.height
        # 카운팅 라인 처리
        self.line_annotate = None 
        self.line_counters = []
        count_line = arg["count_line"]
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
        if arg["is_file"]: # 파일 로직
            self.target_fps = round(self.video_info.fps)
        else: # 스트림 로직
            ffmpeg = FFmpegProcessor(arg["source_video_path"], arg["target_video_path"], arg["pts"], arg["threads"])
            self.process = ffmpeg.setPath()
            self.width = ffmpeg.getWidth()
            self.height = ffmpeg.getHeight()
            self.target_fps = 8 #ffmpeg.getFPS() #검출 후 송출하는 영상 pfs 기준 
        # 영역 처리
        self.detect_zone_show = arg["detect_zone_show"]
        detect_zone_areas = arg["detect_zone_areas"]
        if detect_zone_areas == "": detect_zone_areas = "[]"    
        if not literal_eval(detect_zone_areas): # 전체 영역
            # self.zone_display = False
            self.zones_in = initiate_polygon_zones(
                [np.array([[0,0],[self.width,0],[self.width, self.height],[0,self.height]])], (self.width, self.height), sv.Position.CENTER
            )
        else: # 지정 영역
            # self.zone_display = True
            array = literal_eval(detect_zone_areas)
            temp_array = []
            for arr in array:
                temp_array.append(np.array(arr))
            self.zones_in = initiate_polygon_zones(
                temp_array, (self.width, self.height), sv.Position.CENTER
            )
        # 비영역 처리
        self.detect_none_show = arg["detect_none_show"]
        detect_none_areas = arg["detect_none_areas"]
        if detect_none_areas == "": detect_none_areas = "[]"    
        if literal_eval(detect_none_areas):
            array = literal_eval(detect_none_areas)
            temp_array = []
            for arr in array:
                temp_array.append(np.array(arr))
            self.none_zone_in = initiate_polygon_zones(
                temp_array, (self.width, self.height), sv.Position.CENTER
            )   
        # 패딩 설정
        padding = arg["padding"]
        padding_polygons = set_padding(self.width, self.height, padding)
        for polygon in padding_polygons:
            self.none_zone_in.append(sv.PolygonZone(
                polygon=polygon,
                frame_resolution_wh=(self.width, self.height),
                triggering_position=sv.Position.CENTER
            ))
        
        self.zone_info = dict() # 각 영역(zone)에 해당하는 정보
        # 라벨링
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS, thickness=0)
        self.label_annotator = sv.LabelAnnotator(color=COLORS, text_scale=0.35, text_padding=2, 
            color_lookup=sv.ColorLookup.CLASS, text_color=sv.Color.white()
        )
        # JsonData
        self.none_zone_ids = set()
        self.identity = dict()
        self.detected_identity = dict() # 확인된 identity
        self.frame_number = 0
        self.counting = [] # 패널 카운팅
        self.twins = dict() # 오버레이 관리
        # xml
        self.xml = XML(outPath=arg["xml_output_path"])

    # 비디오 처리
    def process_video(self) -> None: 
        if self.is_file: self.file_process() # file로 로직 처리
        else: 
            producer_process = Process(target=frame_extractor, args=(self.frame_queue, self.source_video_path, 2)) # 마지막 인자값이 몇번째 프레임마다 받을지 결정
            producer_process.start()
            self.stream_process() # rtmp stream으로 처리
            producer_process.join()

    
    def file_process(self) -> None:
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                result = self.model(frame, verbose=False, device=0, imgsz=self.video_info.width, tracker=self.tracker_yaml)[0]
                
                # detection 공통 작업: confidence & nms
                detections = self.common_process(result)

                # Annotation 처리 - frame에 Annotation을 적용함
                annotated_frame = self.annotate_frame(frame, detections)

                sink.write_frame(annotated_frame)
                if self.is_show:
                    cv2.imshow("OpenCV View", annotated_frame)
                if (cv2.waitKey(1) == 27): # ESC > stop
                    break
            print(self.zone_info)
        self.xml.make_xml()


    def stream_process(self) -> None:
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            while True:
                try:
                    frame = self.frame_queue.get(timeout=10)
                except queue.Empty:
                    break
                result = self.model(frame, verbose=False, device=0, imgsz=self.video_info.width, tracker=self.tracker_yaml)[0]
                detections = self.common_process(result)
                annotated_frame = self.annotate_frame(frame, detections)
                numpy_array = np.array(annotated_frame)
                try:
                    if numpy_array is not None:
                        self.process.stdin.write(numpy_array.tobytes())
                        if self.is_show:
                            cv2.imshow("OpenCV View", numpy_array)
                except Exception as e:
                    print(e)
                    self.process.terminate()
                    break
                if cv2.waitKey(1) == 27:  # ESC 키를 눌렀을 때
                    break

                
    def common_process(self, track) -> sv.Detections:
        global now_frame
        self.frame_number += 1 # 현재 프레임 카운팅 - 최초등장 프레임 데이터 필요
        now_frame = self.frame_number
        detections = sv.Detections.from_ultralytics(track)
        detections = detections[detections.confidence > 0.25] 
        detections = detections.with_nms(threshold=0.3) 
        detections = self.tracker.update_with_detections(detections=detections) # 새 번호 발급
        detections = self.detect_in_area(detections) # 영역 처리
        
        # 기록 데이터 세팅
        self.set_identity(detections)
        
        # 기록 데이터 삭제 정리 
        for k, v in list(self.identity.items()):  # .items() 대신 .copy().items() 사용
            delete_flag = False  # 삭제 플래그
            if k in self.identity:
                if self.identity[k]["frame"] + self.keep_frame < self.frame_number: # 마지막 등장 frame이 현재 frame number와의 차이가 keep_frame보다 클경우 삭제
                    delete_flag = True
                
                if self.identity[k]["id"] in self.none_zone_ids: # 비감지 영역에 있는 id의 경우 삭제
                    self.none_zone_ids.remove(self.identity[k]["id"])
                    delete_flag = True
            
                if delete_flag:
                    self.identity.pop(k)
            
            for zone in self.zones_in: # 감지 영역에 걸치면 제거
                for line in zone.polygon:
                    if k not in self.identity:  # 이미 삭제된 경우 스킵
                        continue
                    if k in self.detected_identity and coord.is_line_over(line, self.identity[k]["center"], self.width, self.height):
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
                        if coord.check_overlap(self.identity[tracker_id]["position"], val["position"]) > 0.5: # 오버레이
                            self.twins[tracker_id] = key # 오버레이 관리 (오버레이 대상끼리 id 동기화 작업)
                            target = self.identity[key] # 해당 id는 추적(target)로 관리
                            self.identity[tracker_id]["id"] = target["id"] 
                            self.identity[tracker_id]["class"] = target["class"] # car type number
                            self.identity[tracker_id]["class_type"] = target["class_type"] # car_name
        return detections


    def detect_in_area(self, detections: sv.Detections) -> sv.Detections: # Detect Area 감지 영역 처리
        detections_in_zones = []
        for i, zone_in in enumerate(self.none_zone_in): # 순회하면서 비감지 영역 인덱스 찾아서 제거
            mask = zone_in.trigger(detections=detections)
            detections_in_zone = detections[mask]
            for xyxy in detections_in_zone.xyxy.tolist():
                index = detections.xyxy.tolist().index(xyxy)
                self.none_zone_ids.add(detections.tracker_id[index]) # 비감지 영역에 있는 tracker_id 예측하지 않도록 저장
                detections.tracker_id = np.delete(detections.tracker_id, index, 0)
                detections.xyxy = np.delete(detections.xyxy, index, 0)
                detections.confidence = np.delete(detections.confidence, index, 0)
                detections.class_id = np.delete(detections.class_id, index, 0)
        for i, zone_in in enumerate(self.zones_in):    
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
        
        detections = sv.Detections.merge(detections_in_zones)
        return detections    


    def delete_related_id(self, id) -> None:
        for key, val in self.identity.copy().items(): # 해당 id로 되어있는 데이터 제거
            if val['id'] == id:
                self.identity.pop(key)

    # 화면 표기
    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        detected_ids = detections.tracker_id.tolist()

        # 영역 테두리 처리
        if self.detect_zone_show: # 영역 표기 처리일 때만
            for i, zone_in in enumerate(self.zones_in): # 감지 영역
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
        if self.detect_none_show:
            for i, zone_in in enumerate(self.none_zone_in): # 비감지 영역 
                annotated_frame = sv.draw_polygon(
                    annotated_frame, zone_in.polygon, sv.Color.from_hex('#ec4141')
                )
                sv.PolygonZoneAnnotator(
                    zone=zone_in,
                    color=sv.Color.from_hex('#ec4141'),
                    thickness=0.1,
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

        displayed_ids= set()
        # 차량 라벨
        for key, val in self.identity.items():
            # if key not in cant:
            if self.identity[key]['id'] in displayed_ids:
                continue
            displayed_ids.add(self.identity[key]['id'])
            label = f"[{self.identity[key]['id']}] {self.identity[key]['speed']}km"

            # 처리 후 self.savedJson에 JSON 데이터로 저장함
            if self.is_file:
                position = " ".join([
                    str(int(self.identity[key]["position"][0])), 
                    str(int(self.identity[key]["position"][1])), 
                    str(int(self.identity[key]["position"][2])), 
                    str(int(self.identity[key]["position"][3]))])
                self.xml.save_data({
                    "card": self.identity[key]["id"], 
                    "now_frame": self.identity[key]["frame"], 
                    "class": self.identity[key]["class_type"], 
                    "position": position})

            plot_one_box2(
                self.identity[key]["position"], 
                annotated_frame, 
                self.identity[key]["class_type"], 
                label=label, 
                color=colors_bgr[self.identity[key]["class"]], 
                line_thickness=1
            )
                
        # 차량 패널 표시
        self.set_info_panel(annotated_frame)

        return annotated_frame
    
    # 검출 정보로 JSON 데이터 수집
    def set_identity(self, detections: sv.Detections) -> None:
        
        none_detected_tracker_ids = list(set(self.detected_identity)-set(detections.tracker_id))
        # 비검출 데이터 처리
        for tracker_id in none_detected_tracker_ids:
            if tracker_id not in self.identity:
                continue # key 있는 경우에만
            if tracker_id in self.none_zone_ids:
                continue
            xyxy = coord.predict_xyxy(
                xyxys=self.identity[tracker_id]['position_array'], 
                gaps=self.identity[tracker_id]['position_gap'],
                direct=self.identity[tracker_id]['direct']
            )
            self.set_position(tracker_id, xyxy)
            self.identity[tracker_id]['predicted'] = True
        # 검출 데이터 처리
        for i in range(len(detections.xyxy)):
            xyxy = detections.xyxy[i]
            tracker_id = detections.tracker_id[i]
            center = (round((xyxy[0]+xyxy[2])/2, 2), round((xyxy[1]+xyxy[3])/2,2))
            if tracker_id not in self.identity: # 처음으로 검출된 tracker_id / self.identity[tracker_id] 추가
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
                    "start_time": timer.current, # 비추적
                    "now_time": timer.current,
                    "predicted" : False
                }
            else: # 기존에 검출된 기록이 있는 tracker_id의 경우
                self.identity[tracker_id]['frame'] = self.frame_number
                self.set_position(tracker_id, xyxy)
                self.identity[tracker_id]['predicted'] = False
                if self.identity[tracker_id]['speed'] == 0 and len(self.identity[tracker_id]['center_array']) > 2: # 속도가 0이면 즉시 속도 계산
                    speed = coord.estimatespeed(self.identity[tracker_id]['center_array'][-1], self.identity[tracker_id]['center_array'][-2], self.target_fps)
                    self.identity[tracker_id]['speed'] = speed
                elif self.frame_number > timer.before_frame + timer.work_frame : # 작업 프레임 단위 (영상 1초)마다 아래 갱신
                    self.identity[tracker_id]['now_time'] = timer.current
                    if len(self.identity[tracker_id]['center_array']) > 2:
                        speed = coord.estimatespeed(self.identity[tracker_id]['center_array'][-1], self.identity[tracker_id]['center_array'][-2], self.target_fps)
                        self.identity[tracker_id]['speed'] = speed
    
    # 공통 좌표 구하기                    
    def set_position(self, tracker_id:int, xyxy:list) -> None:
        # xyxy 최대 범위는 자기 해상도까지
        xyxy[0] = 0 if xyxy[0] < 0 else self.width if xyxy[0] > self.width else xyxy[0]
        xyxy[1] = 0 if xyxy[1] < 0 else self.height if xyxy[1] > self.height else xyxy[1]
        xyxy[2] = 0 if xyxy[2] < 0 else self.width if xyxy[2] > self.width else xyxy[2]
        xyxy[3] = 0 if xyxy[3] < 0 else self.height if xyxy[3] > self.height else xyxy[3]
        center = (round((xyxy[0]+xyxy[2])/2, 2), round((xyxy[1]+xyxy[3])/2,2))
        self.identity[tracker_id]['position'] = xyxy
        self.identity[tracker_id]['position_array'].append(xyxy)
        self.identity[tracker_id]['position_gap_array'].append(coord.gap(self.identity[tracker_id]["position_array"][-1], self.identity[tracker_id]["position_array"][-2]))
        self.identity[tracker_id]['position_gap'] = coord.avg_weight_gap(self.identity[tracker_id]['position_gap_array'])
        self.identity[tracker_id]['center'] = center
        self.identity[tracker_id]['center_array'].append(center)
        self.identity[tracker_id]['direct'] = coord.direct(self.identity[tracker_id]["position_array"]) # 방향
        
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
    # timer.current_time() # 시간 측정 시작
    processor = VideoProcessor(arg.get_argment())
    processor.process_video()
    # timer.Timer.cancel() # 타임쓰레드 반드시 종료