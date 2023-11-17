import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

import subprocess
import ffmpeg 

rtmp_url = "rtmp://localhost:1935/live/mist2"
path = "rtmp://localhost:1935/live/mist1"

# scene 여러 개가 안됨
def main():    

    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    process1 = (
        ffmpeg
        .input(path)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', vframes=8)
        .run_async(pipe_stdout=True)
    )

    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(rtmp_url, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    
    # print(process1)
    
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            continue           
            
        in_frame = (
            np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        )

        
        model = YOLO("yolov8l.pt")
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )
        for result in model.track(source=in_frame, show=True):
        
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
            # return numpy_array
            process2.stdin.write(
                numpy_array
                .astype(np.uint8)
                .tobytes()
            )


        
    # process2.stdin.close()
    # process1.wait()
    # process2.wait()
    

if __name__ == "__main__":
    main()