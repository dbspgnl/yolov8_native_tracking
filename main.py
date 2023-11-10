import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

import subprocess

def main():    
    rtmp_url = "rtmp://localhost:1935/live/mist2"
    path = "rtmp://localhost:1935/live/mist1"
    cap = cv2.VideoCapture(path)
    # cap = cv2.VideoCapture(0)

    # gather video info to ffmpeg
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = 5
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
            '-vf', 'scale=180:120',
            '-f', 'flv',
            rtmp_url]

    p = subprocess.Popen(command, stdin=subprocess.PIPE)
    
    # =======================================================================================
    
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    model = YOLO("yolov8l.pt")
    for result in model.track(source=path, show=False, stream=True):
    
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        detections = detections[detections.class_id != 60] # 다이닝 테이블 제외
        
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
        
        numpy_array = np.array(frame)
        p.stdin.write(numpy_array.tobytes()) # rtmp 송신
        cv2.imshow("yolov8", numpy_array)

        if (cv2.waitKey(30) == 27):
            break
        
if __name__ == "__main__":
    main()