import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 定义视频格式
vw = cv2.VideoWriter("output.mp4", fourcc, 25, (640, 480))  # 创造一个视频文件

cv2.namedWindow("new", cv2.WINDOW_NORMAL)
cv2.namedWindow("test",cv2.WINDOW_NORMAL)
cv2.resizeWindow("new", 640, 480)  # Resize the window to match the video frame size

video = cv2.VideoCapture(0)#获取视频

def videoproses(src):
    img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(15,15),sigmaX=5,sigmaY=5)
    cv2.imshow("test",img)

while video.isOpened():
    ret, frame = video.read()#读取视频帧
    if not ret:
        print("Error: Could not read frame.")
        break
    
    if frame.shape[1] != 640 or frame.shape[0] != 480:
        frame = cv2.resize(frame, (640, 480))

    frame = cv2.flip(frame,1)#水平翻转
    videoproses(frame)

    cv2.imshow("new", frame)
    #vw.write(frame)#写入文件
    key = cv2.waitKey(40) & 0xFF
    if key == ord('q'):
        break

video.release()
vw.release()
cv2.destroyAllWindows()