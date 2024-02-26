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
from datetime import datetime
from ast import literal_eval
from tqdm import tqdm
from multiprocessing import freeze_support

car_names = ['car', 'truck', 'bus', 'vehicle'] # 차량 info 종류
colors_bgr = [[17, 133, 254],[255, 156, 100],[11, 189, 128],[0, 255, 255]] # 차량 info bgr 색상
car_colors = ['#ff8a0d', '#5d9bff', '#7cae01', '#ffeb00'] # 차량 라벨 색상
COLORS = sv.ColorPalette.from_hex(car_colors) # car, truck, bus, vehicle

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


class FFmpegProcessor:
    def __init__(self, input_path, output_path) -> None:
        self.cap = cv2.VideoCapture(input_path)
        self.target:str = output_path
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
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
                '-sws_flags', 'lanczos',
                '-filter:v', 'setpts=4.0*PTS',
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
    ) -> None:
        # 인자값 설정
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.is_show = is_show
        self.is_file = is_file
        self.is_count_show = is_count_show
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
            ffmpeg = FFmpegProcessor(source_video_path, target_video_path)
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

        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS, thickness=0)
        self.label_annotator = sv.LabelAnnotator(color=COLORS, text_scale=0.35, text_padding=2, color_lookup=sv.ColorLookup.CLASS)
        self.identity = dict()
        self.frame_number = 0
        self.counting = [] # 패널 카운팅


    # 비디오 처리
    def process_video(self): 
        if self.is_file: self.file_process() # file로 로직 처리
        else: self.stream_process() # rtmp stream으로 처리
    
    
    def file_process(self):
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                result = self.model(frame, verbose=False, device=0, imgsz=self.video_info.width)[0]

                now = datetime.now()
                timestamp = now.timestamp()
                format_time = now.strftime('%Y-%m-%d %H:%M:%S)')
                self.frame_number += 1 # 현재 프레임 카운팅
                
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.confidence > 0.3] # 정확도 0.3 이상만
                detections = detections.with_nms(threshold=0.7) # 비최대 억제 0.7
                detections = self.detect_in_area(detections) # 영역만 디텍팅
                detections = self.tracker.update_with_detections(detections=detections)

                # Annotation 처리
                annotated_frame = self.annotate_frame(frame, detections)

                sink.write_frame(annotated_frame)
                if self.is_show:
                    cv2.imshow("OpenCV View", annotated_frame)
                if (cv2.waitKey(1) == 27): # ESC > stop
                    break
            print(self.zone_info)


    def stream_process(self):    
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for result in self.model.track(source=self.source_video_path, show=False, stream=True, device=0, verbose=False, agnostic_nms=True, imgsz=1920):
                if result.boxes.id is None: # 검출이 안되면 스킵
                    continue
                frame = result.orig_img

                now = datetime.now()
                timestamp = now.timestamp()
                format_time = now.strftime('%Y-%m-%d %H:%M:%S)')
                self.frame_number += 1 # 현재 프레임 카운팅
                
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.confidence > 0.3] # 정확도 0.3 이상만
                detections = detections.with_nms(threshold=0.7) # 비최대 억제 0.7
                detections = self.detect_in_area(detections) # 영역만 디텍팅
                detections = self.tracker.update_with_detections(detections=detections) # 새 번호 발급

                # Tracker id
                if result.boxes is None or result.boxes.id is None:
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                    
                # Annotation 처리
                annotated_frame = self.annotate_frame(frame, detections)
                # Output 처리
                numpy_array = np.array(annotated_frame)
                
                if numpy_array is not None:
                    self.process.stdin.write(numpy_array.tobytes()) # Output FFmepg
                    if self.is_show:
                        cv2.imshow("OpenCV View", numpy_array)
                
                if (cv2.waitKey(1) == 27): # ESC > stop
                    break
    def detect_in_area(self, detections: sv.Detections):
        detections_in_zones = [] # Detect Area 감지 영역 처리
        for i, zone_in in enumerate(self.zones_in):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)        
        detections = sv.Detections.merge(detections_in_zones)
        return detections

    # 화면 표기
    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        
        # 기록 데이터 삭제 정리
        for k,v in self.identity.copy().items():
            if k not in detections.tracker_id:
                self.identity.pop(k)
        
        # 기록 데이터 세팅
        self.set_identity(detections)
        
        # 차량 카운팅
        self.counting = self.set_counting(detections)

        # 영역 테두리 처리
        if self.zone_display: # 영역 표기 처리일 때만
            for i, zone_in in enumerate(self.zones_in):
                annotated_frame = sv.draw_polygon(
                    annotated_frame, zone_in.polygon, sv.Color.from_hex('#ffffff')
                )
                sv.PolygonZoneAnnotator(
                    zone=zone_in,
                    color=sv.Color.from_hex('#ffffff'),
                    thickness=1,
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
            
        # 오브젝트 바운딩 박스
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        # 오브젝트 라벨
        labels = [
            f"{self.identity[tracker_id]['id']} : {self.identity[tracker_id]['speed']}km"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        
        # 프레임 라벨 처리
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        
        # 차량 패널 표시
        self.set_info_panel(annotated_frame)

        return annotated_frame
    
    # 검출 정보로 JSON 데이터 수집
    def set_identity(self, detections: sv.Detections) -> None:
        for i in range(len(detections.xyxy)):
            xyxy = detections.xyxy[i]
            tracker_id = detections.tracker_id[i]
            frame_number = self.frame_number
            center = (round((xyxy[0]+xyxy[2])/2, 2), round((xyxy[1]+xyxy[3])/2,2))
            if tracker_id not in self.identity:
                self.identity[tracker_id] = {
                    "id": tracker_id,
                    "position": xyxy,
                    "frame": frame_number,
                    "center": center,
                    "center_array": [center],
                    "speed": 0
                }
            else:
                self.identity[tracker_id]['center_array'].append(center)
                if len(self.identity[tracker_id]['center_array']) > 2:
                    speed = self.estimatespeed(self.identity[tracker_id]['center_array'][-1], self.identity[tracker_id]['center_array'][-2])
                    self.identity[tracker_id]['speed'] = speed
                else:
                    self.identity[tracker_id]['speed'] = "-"

    # 좌표값으로 속도 계산
    def estimatespeed(self, Location1, Location2):
        location_x = abs(Location2[0] - Location1[0])
        location_y = abs(Location2[1] - Location1[1])
        d_pixel = math.sqrt(math.pow(location_x, 2) + math.pow(location_y, 2))
        d_meters = (d_pixel/6.75) # 1080 해상도 거리 160m = 미터당 6.75픽셀
        speed = d_meters * self.target_fps * 3.6 # meter x fps x km
        return int(speed)
    
    # 차량 수 카운팅
    def set_counting(self, detections):
        count = []
        for i in range(len(car_names)):
            count.append(0)
        for id in detections.class_id:
            count[id] += 1
        return count
        
    # 패널 info 표시
    def set_info_panel(self, frame):
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
    parser = argparse.ArgumentParser(description="Traffic Flow Analysis with YOLOv8")
    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--zone_in_polygons",
        default=None,
        help="Coordinate array for detection area",
        type=str,
    )
    parser.add_argument(
        "--count_line",
        default=None,
        help="Line zone coordinates for entry/exit counting processing",
        type=str,
    )
    parser.add_argument(
        "--count_show",
        default=False,
        help="Line zone Count Label Show",
        type=str,
    )
    parser.add_argument(
        "--show",
        default=False,
        help="OpenCV Show",
        type=str,
    )
    parser.add_argument(
        "--file",
        default=False,
        help="Make File",
        type=str,
    )
    args = parser.parse_args()
    count_show_bool_true = (args.count_show == 'true') # str > bool
    show_bool_true = (args.show == 'true') # str > bool
    file_bool_true = (args.file == 'true') # str > bool
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        zone_in_polygons=args.zone_in_polygons,
        count_line=args.count_line,
        is_count_show=count_show_bool_true,
        is_show=show_bool_true,
        is_file=file_bool_true,
    )
    processor.process_video()