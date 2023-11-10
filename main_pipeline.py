import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

import subprocess
import ffmpeg # 여기 오면 바로 ffmpeg 진입함...

def main():    

    process1 = (
        ffmpeg
        .input("rtmp://localhost:1935/live/mist1")
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=8)
        .run_async(pipe_stdout=True)
    )

    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output("rtmp://localhost:1935/live/mist2", pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )


    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )

        # See examples/tensorflow_stream.py:
        # out_frame = deep_dream.process_frame(in_frame)
        out_frame = deep_dream(in_frame)

        process2.stdin.write(
            out_frame
            .astype(np.uint8)
            .tobytes()
        )

    process2.stdin.close()
    process1.wait()
    process2.wait()

def deep_dream(path):
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
        return numpy_array
        # p.stdin.write(numpy_array.tobytes()) # rtmp 송신
        # while cap.isOpened():
        # for _ in range(len(frame)):
        #     p.stdin.write(numpy_array.tobytes()) # rtmp 송신
        
        # cv2.imshow("yolov8", numpy_array)

        # if (cv2.waitKey(30) == 27):
        #     break
    

if __name__ == "__main__":
    main()