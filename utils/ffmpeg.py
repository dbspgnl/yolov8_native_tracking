import subprocess
import cv2

class FFmpegProcessor:
    def __init__(self, input_path:str, output_path:str, pts:str, threads:str) -> None:
        self.cap = cv2.VideoCapture(input_path)
        self.target:str = output_path
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pts = pts
        self.threads = threads
        
    def getWidth(self): return self.width
    def getHeight(self): return self.height
    def getFPS(self): return self.fps
    
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
                '-filter:v', 'setpts={}*PTS'.format(self.pts), 
                '-threads', self.threads,
                '-r', '30',
                '-f', 'flv',
                self.target]
        return subprocess.Popen(command, stdin=subprocess.PIPE)