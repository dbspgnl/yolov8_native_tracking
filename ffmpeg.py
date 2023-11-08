import subprocess
import cv2
rtmp_url = "rtmp://127.0.0.1:1935/live/mist2"

# In my mac webcamera is 0, also you can set a video file name instead, for example "/home/user/demo.mp4"
path = 0
# path = "35s.mp4"
# path = "http://localhost:8080/hls/mist1/index.m3u8"
cap = cv2.VideoCapture(path)

# gather video info to ffmpeg
fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = 640
height = 480

# command and params for ffmpeg
command = ['ffmpeg',
		'-y',
		'-f', 'vfwcap',
		'-video_size', "{}x{}".format(width, height),
		'-framerate', '25',
		'-i', '/dev/video0',
		'-vcodec', 'h264',
		'-c:a', 'aac',
		'-ar', '44100',
		'-f', 'flv',
		rtmp_url]

# using subprocess and pipe to fetch frame data
p = subprocess.Popen(command, stdin=subprocess.PIPE)


while cap.isOpened():
    if not ret:
        print("frame read failed")
        break
    ret, frame = cap.read()

    # YOUR CODE FOR PROCESSING FRAME HERE

    # write to pipe
    p.stdin.write(frame.tobytes())