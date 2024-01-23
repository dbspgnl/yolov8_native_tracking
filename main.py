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
# from collections import defaultdict, deque

COLORS = sv.ColorPalette.from_hex(['#ff8a0d', '#5d9bff', '#7cae01', '#ffeb00']) # car, truck, bus, vehicle

ZONE_IN_POLYGONS = [
    np.array([
        [87, 532],[1783, 532],[1783, 812],[87, 812]
    ])
]


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
        
    def getWidth(self):
        return self.width
    
    def getHeight(self):
        return self.height
    
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
                # '-filter:v', 'minterpolate=mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=30',
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
        target_video_path: str = None,
    ) -> None:
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()
        # self.tracker = sv.ByteTrack( frame_rate=60, track_thresh=0.3 )
        
        ffmpeg = FFmpegProcessor(source_video_path, target_video_path)
        self.process = ffmpeg.setPath()
        self.width = ffmpeg.getWidth()
        self.height = ffmpeg.getHeight()
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, (self.width, self.height), sv.Position.CENTER
        )
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(color=COLORS, thickness=0)
        self.label_annotator = sv.LabelAnnotator(color=COLORS, text_scale=0.35, text_padding=2, color_lookup=sv.ColorLookup.CLASS)
        # self.overlay = ImageOverlayBlending("1920.png") # 블렌딩 이미지 경로
        self.identity = dict()
        self.frame_number = 0
        self.video_info = sv.VideoInfo.from_video_path(video_path=self.source_video_path)
        # self.coordinates = defaultdict(lambda: deque(maxlen=self.video_info.fps))

    def process_video(self): 
        second = 0
        # video_info = sv.VideoInfo.from_video_path(video_path=self.source_video_path)
        # print(video_info.fps)
        # coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
        
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for result in self.model.track(source=self.source_video_path, show=False, stream=True, device=0, verbose=False, agnostic_nms=True, imgsz=1920):
                now = datetime.now()
                timestamp = now.timestamp()
                format_time = now.strftime('%Y-%m-%d %H:%M:%S)')
                self.frame_number += 1 # 현재 프레임 카운팅
                
                # if now.second is not second: # 초당 프레임 진행도 측정
                #     second = now.second
                #     print(f"{second}초")
                #     print(self.frame_number)
                
                if result.boxes.id is None: # 검출이 안되면 스킵
                    continue
                frame = result.orig_img
                detections = sv.Detections.from_ultralytics(result)
                #
                detections = detections[detections.confidence > 0.3] # 정확도 0.3 이상만
                detections = detections.with_nms(threshold=0.7) # 
                detections = self.tracker.update_with_detections(detections=detections)
                #

                # points = detections.get_anchor_coordinates(
                #     anchor=sv.Position.CENTER
                # )
                # for tracker_id, [_, y] in zip(detections.tracker_id, points):
                #     self.coordinates[tracker_id].append(y)

                
                
                # Tracker id
                if result.boxes is None or result.boxes.id is None:
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

                # Annotation 처리
                annotated_frame = self.annotate_frame(frame, detections)
                # Output 처리
                numpy_array = np.array(annotated_frame)
                # 오버레이 블렌딩 처리
                # numpy_array = self.overlay.setMix(numpy_array)
                
                if numpy_array is not None:
                    self.process.stdin.write(numpy_array.tobytes()) # Output FFmepg
                    cv2.imshow("OpenCV View", numpy_array)
                
                if (cv2.waitKey(1) == 27): # ESC > stop
                    break

            
    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        
        self.set_identity(detections)
        
        # Detect Area 감지 영역 처리
        detections_in_zones = []
        for zone_in in self.zones_in:
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
        detections = sv.Detections.merge(detections_in_zones)

        # 영역 테두리 처리
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
            
        # 오브젝트 바운딩 박스
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        # 오브젝트 라벨
        labels = [
            # f"{tracker_id} {self.model.names[class_id]} {confidence:0.2f}"
            # f"{tracker_id} : {self.identity[tracker_id]['speed']}km" if self.identity[tracker_id] and self.identity[tracker_id]['speed'] is not None else f"{tracker_id}"
            f"{tracker_id} : {self.identity[tracker_id]['speed']}km"
            # f"{self.identity[tracker_id]['id']} : {self.identity[tracker_id]['speed']}km"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        
        # labels = []
        # for tracker_id in detections.tracker_id:
        #     if len(self.coordinates[tracker_id]) < self.video_info.fps / 2:
        #         labels.append(f"#{tracker_id}")
        #     else:
        #         coordinate_start = self.coordinates[tracker_id][-1]
        #         coordinate_end = self.coordinates[tracker_id][0]
        #         distance = abs(coordinate_start - coordinate_end)
        #         time = len(self.coordinates[tracker_id]) / self.video_info.fps
        #         speed = distance / time * 3.6
        #         labels.append(f"#{tracker_id} {int(speed)} km/h")    

        # 프레임 라벨 처리
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        return annotated_frame
    
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
                if len(self.identity[tracker_id]['center_array']) > 8:
                    speed = self.estimatespeed(self.identity[tracker_id]['center_array'][-1], self.identity[tracker_id]['center_array'][-8])
                    self.identity[tracker_id]['speed'] = speed
                else:
                    self.identity[tracker_id]['speed'] = "-"

    
    def estimatespeed(self, Location1, Location2):
        location_x = abs(Location2[0] - Location1[0])
        location_y = abs(Location2[1] - Location1[1])
        d_pixel = math.sqrt(math.pow(location_x, 2) + math.pow(location_y, 2))
        ppm = (1/6.66) # 0.15 <-- 현재 고정값 60/9
        d_meters = d_pixel/(1/ppm) # defining thr pixels per meter
        # speed = d_meters * 8.6 * 3.6 # meter x fps x km
        speed = d_meters * 3.6 # meter x fps x km
        return int(speed)
    
    
if __name__ == "__main__":
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
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
    )
    processor.process_video()