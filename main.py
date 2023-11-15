import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import subprocess
import torch
import torchvision

rtmp_url = "rtmp://localhost:1935/live/mist2"
path = "rtmp://localhost:1935/live/mist1" # webcam:0 / rtmp 없으면 argument error
LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)

def tracking(p):
    model = YOLO("best.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    # 소스, 보기, 스트림, GPU, 로그, 더블디텍션
    for result in model.track(source=path, show=False, stream=True, device=0, verbose=False, agnostic_nms=True):
    
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        detections = detections[detections.class_id != 60] # 다이닝 테이블 60번 제외
        
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        
        numpy_array = np.array(frame)
        p.stdin.write(numpy_array.tobytes()) # rtmp 송신
        
        # cv2.imshow("OpenCV View", numpy_array)
        # if (cv2.waitKey(30) == 27):
        #     break
    
    
def main():    
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    command = ['ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "{}x{}".format(width, height),
            '-r', str(fps),
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
            rtmp_url]

    p = subprocess.Popen(command, stdin=subprocess.PIPE)
    tracking(p)

        
if __name__ == "__main__":
    main()