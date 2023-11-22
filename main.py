import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import subprocess
import torch
import torchvision
import argparse

colors = sv.ColorPalette.default()
polygons = [
    np.array([
        [300, 300],[600, 300],[600, 600],[300, 600]
    ])
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
                '-vf', 'scale=640:480',
                '-filter:v', 'minterpolate=mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=60',
                '-filter:v', 'setpts=4.0*PTS',
                '-r', '15',
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
        
        ffmpeg = FFmpegProcessor(source_video_path, target_video_path)
        self.process = ffmpeg.setPath()
        self.width = ffmpeg.getWidth()
        self.height = ffmpeg.getHeight()

    def process_video(self): #1

        zones = [
            sv.PolygonZone(
                polygon=polygon,
                # frame_resolution_wh=video_info.resolution_wh
                frame_resolution_wh=(self.width,self.height)
            )
            for polygon
            in polygons
        ]
        zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=colors.by_idx(index),
                thickness=4,
                text_thickness=8,
                text_scale=4
            )
            for index, zone
            in enumerate(zones)
        ]
        box_annotators = [
            sv.BoxAnnotator(
                color=colors.by_idx(index),
                thickness=4,
                text_thickness=4,
                text_scale=2
                )
            for index
            in range(len(polygons))
        ]
        
        # 소스, 보기, 스트림, GPU, 로그, 더블디텍션
        for result in self.model.track(source=self.source_video_path, show=False, stream=True, device=0, verbose=False, agnostic_nms=True):
        
            frame = result.orig_img
            detections = sv.Detections.from_yolov8(result)
            
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            detections = detections[detections.class_id != 60] # 다이닝 테이블 60번 제외
            
            labels = [
                f"{tracker_id} {self.model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            
            for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
                frame = zone_annotator.annotate(scene=frame)
            
            numpy_array = np.array(frame)
            self.process.stdin.write(numpy_array.tobytes()) # rtmp 송신
            
            cv2.imshow("OpenCV View", numpy_array)
            if (cv2.waitKey(30) == 27):
                break
    
        
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