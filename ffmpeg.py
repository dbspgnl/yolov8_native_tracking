import subprocess
import cv2
rtmp_url = "rtmp://localhost:1935/live/mist2"

# In my mac webcamera is 0, also you can set a video file name instead, for example "/home/user/demo.mp4"
# path = 0
path = "rtmp://localhost:1935/live/mist1"
cap = cv2.VideoCapture(path)

# gather video info to ffmpeg
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("fps: "+str(fps))
print("width: "+str(width))
print("width: "+str(height))

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
		'-f', 'flv',
		rtmp_url]

p = subprocess.Popen(command, stdin=subprocess.PIPE)

# opencv show
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     # dst = cv2.cvtColor(frame, cv2.COLOR_BGR2)

#     # Display the resulting frame
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

while cap.isOpened():
    # if not ret:
    #     print("frame read failed")
    #     break
    ret, frame = cap.read()

    # YOUR CODE FOR PROCESSING FRAME HERE

    # write to pipe
    p.stdin.write(frame.tobytes())
    # print(frame.tobytes())


# cap.release()
# cv2.destroyAllWindows()