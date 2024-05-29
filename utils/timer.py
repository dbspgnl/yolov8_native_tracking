import threading
import datetime

current = datetime.datetime.now()
Timer:threading.Timer = None
before_frame:int = 0
now_frame:int = 0
work_frame:int = 60

def current_time() -> None: # 시간 측정 기능
    global current, Timer, before_frame, now_frame, work_frame
    work_frame = now_frame - before_frame # 실제 프레임 (작업 프레임) 설정
    before_frame = now_frame
    Timer = threading.Timer(1, current_time) # 1초마다 재호출
    Timer.start() 