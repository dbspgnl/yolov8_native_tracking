<h1 align="center">YOLOv8 LIVE</h1>

<!-- <p align="center">
    <a href="https://youtu.be/QV85eYOb7gk">
        <img src="https://user-images.githubusercontent.com/26109316/218765786-5ae9d65d-10fc-4626-af72-8e833e3b8f34.jpg" alt="count-objects-crossing-line">
    </a>
</p> -->

## 👋 hello

A short script showing how to build simple real-time video analytics apps using [YOLOv8](https://github.com/ultralytics/ultralytics) and [Supervision](https://github.com/roboflow/supervision). Try it out, and most importantly have fun! 🤪

## 💻 Install

```bash
# create python virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## 📸 Execute

```bash
python3 -m main
```

## Command
```bash
python traffic.py --source_weights_path data/traffic_analysis.pt --source_video_path data/traffic_analysis.mov --confidence_threshold 0.3 --iou_threshold 0.5 --target_video_path data/traffic_analysis_result.mov

python traffic.py --source_weights_path best.pt --source_video_path rtmp://localhost:1935/live/mist1 --confidence_threshold 0.3 --iou_threshold 0.5 --target_video_path data/traffic_analysis_result.mp4

```