import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import subprocess
import torch
import torchvision
import argparse
from typing import Dict, List, Set, Tuple

COLORS = sv.ColorPalette.default()
ZONE_IN_POLYGONS = [
    np.array([
        [87, 532],[1783, 532],[1783, 612],[87, 612]
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

class ImageOverlayBlending:
    def __init__(self, imgPath) -> None:
        self.imgPath:str = imgPath
        self.mix:float = 0
        cv2.namedWindow("OpenCV View") #
        cv2.createTrackbar('Mixing', 'OpenCV View', 0,100, lambda x:x) 
    
    def setMix(
        self,
        numpy_array
    ) -> np.array:
        numpy_array = cv2.resize(numpy_array, dsize=(1920, 1080), interpolation=cv2.INTER_AREA) #
        im2 = cv2.imread(self.imgPath) #
        im2 = cv2.resize(im2, dsize=(1920, 1080), interpolation=cv2.INTER_AREA) #
        img = cv2.addWeighted(numpy_array, float(100-self.mix)/100, im2 , float(self.mix)/100, 0) #
        self.mix = cv2.getTrackbarPos('Mixing','OpenCV View')
        return img
    
    
class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
    ) -> sv.Detections:
        detections_all = sv.Detections.merge(detections_in_zones) # 영역 내 데이터만 처리
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            if detections_in_zone.tracker_id is not None:
                for tracker_id in detections_in_zone.tracker_id:
                    self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
        
        detections_all.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(x, -1)
        )(detections_all.tracker_id)
        return detections_all[detections_all.class_id != -1]


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
        
        ffmpeg = FFmpegProcessor(source_video_path, target_video_path)
        self.process = ffmpeg.setPath()
        self.width = ffmpeg.getWidth()
        self.height = ffmpeg.getHeight()
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, (self.width, self.height), sv.Position.CENTER
        )
        self.detections_manager = DetectionsManager()
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=0)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.overlay = ImageOverlayBlending("1920.png") # 블렌딩 이미지 경로

    def process_video(self): 
        
        for result in self.model.track(source=self.source_video_path, show=False, stream=True, device=0, verbose=False, agnostic_nms=True, imgsz=1920):
            frame = result.orig_img
            detections = sv.Detections.from_ultralytics(result)
            # Tracker id
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            # Detect Area 처리
            detections_in_zones = []
            for zone_in in self.zones_in:
                detections_in_zone = detections[zone_in.trigger(detections=detections)]
                detections_in_zones.append(detections_in_zone)
            detections = self.detections_manager.update(
                detections, detections_in_zones
            )
            # Annotation 처리
            annotated_frame = self.annotate_frame(frame, detections)
            # Output 처리
            numpy_array = np.array(annotated_frame)
            # 오버레이 블렌딩 처리
            numpy_array = self.overlay.setMix(numpy_array)
            
            if numpy_array is not None:
                self.process.stdin.write(numpy_array.tobytes()) # Output FFmepg
                cv2.imshow("OpenCV View", numpy_array)
            
            if (cv2.waitKey(30) == 27): # ESC > stop
                break
            
    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        # 영역 처리
        for i, zone_in in enumerate(self.zones_in):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i+1]
            )
            sv.PolygonZoneAnnotator(
                zone=zone_in,
                color=COLORS.colors[i+1],
                thickness=1,
                text_thickness=1,
                text_scale=1
            )
        # 바운딩 박스
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        # 라벨
        labels = [
            f"{tracker_id} {self.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        return annotated_frame
    
        
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