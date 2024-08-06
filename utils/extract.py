from multiprocessing import Queue, Event
import cv2

def frame_extractor(queue: Queue, source_video_path: str, skip_frames: int):
    cap = cv2.VideoCapture(source_video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % skip_frames == 0:
            queue.put(frame)
        frame_count += 1
    cap.release()