import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

def main():    
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    model = YOLO("yolov8l.pt")
    for result in model.track(source=0, show=True, stream=True):
    
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        detections = detections[detections.class_id != 60] # 다이닝 테이블 제외
        
        labels = [
            f"{int(tracker_id)} {model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break
        
if __name__ == "__main__":
    main()