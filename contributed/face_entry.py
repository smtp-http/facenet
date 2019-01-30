import cv2
import dlib
import os
import sys
from scipy import misc
import time
from multiprocessing import Process, Queue
from random import random
import time
import glob
import facenet.src.align.detect_face as detect_face
import numpy as np


#tasks_queue = JoinableQueue()
results_queue = Queue()
convert_queue = Queue()
message_queue = Queue()

image_extension = ".jpg"
target_img_size = 160

IsCatchVideoQuit = False
IsFaceDectecdQuit = False


name = input('please input your name:')
window_name = "face picture entry"
src_path,_ = os.path.split(os.path.realpath(__file__))
print(src_path)
data_dir = os.path.join(src_path,"data")
print(data_dir)
output_dir = os.path.join(data_dir,name)
#size = 64

img_dir = os.path.join(src_path,"images")
size_dir = os.path.join(img_dir,"%d" % target_img_size)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)



#使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()

#videoName = './' + name + '.avi'
videoName = "rtsp://admin:ABC_123456@172.17.208.150:554/Streaming/Channels/101?transportmode=unicast"
# 打开摄像头 参数为输入流，可以为摄像头或视频文件


def CatchUsbVideo(out_queue,msg_q):
    process_flag = False
    cv2.namedWindow(window_name)
 
    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(videoName)      
    n = 1
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:            
            break                    
 
        #显示图像并等待10毫秒按键输入，输入‘q’退出程序
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) #cv2.WND_PROP_FULLSCREEN)
        #cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
        n = n + 1
        if n == 30:
            if process_flag == True:
                out_queue.put(frame)
            n = 1
            
        else:
            frameresize = cv2.resize(frame,(1280,800))
            cv2.imshow(window_name, frameresize)

        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
        if c & 0xFF == ord('d'):
            process_flag = True
        if not msg_q.empty():
            if msg_q.get() == "complate_msg":
                break


 
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows() 


def SendImg(img):
    print(img)

add = 30
def FaceDectection(in_queue,msg_q):
    index = 0
    num = 0
    temporary_dir = os.path.join(output_dir,"%d" % index)
    if  not os.path.exists(temporary_dir):
        os.makedirs(temporary_dir)
    while 1:
        img = in_queue.get()
        g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(g_img, 1)

        for i, d in enumerate(dets):
            print("----- i:%d\n" % i)
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            print("index: %d   num %d " % (index,num))


            face = img[x1-add:y1 + add,x2-add:y2 + add]


            if face.shape[0] > 250 :  # check picture size
                g_img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                lm = cv2.Laplacian(g_img, cv2.CV_64F).var()
                print("## lm : %d \n" % lm)
                if lm < 300:
                    continue
                file = os.path.join(temporary_dir,str(num)+image_extension)
                cv2.imwrite(file, face)
                #out_convert_queue.put(file)
                num += 1
         

        if num >= 5:
            img_file = os.path.join(temporary_dir,"*" + image_extension)
            print(img_file)
            img_name_list = glob.glob(img_file)

            for nm in img_name_list:
                SendImg(nm)

                #out_convert_queue.put(nm)
            index += 1
            temporary_dir = os.path.join(output_dir,"%d" % index)
            if  not os.path.exists(temporary_dir):
                os.makedirs(temporary_dir)
            num = 0

        if not msg_q.empty():
            if msg_q.get() == "complate_msg":
                break


#if __name__ == '__main__':
#    CatchUsbVideo()
processes = []

p = Process(target=CatchUsbVideo, args=(results_queue,message_queue,))
p.start()
processes.append(p)

p = Process(target=FaceDectection, args=(results_queue,convert_queue,message_queue,))
p.start()
processes.append(p)



for p in processes:
    p.join()

while 1:
    time.sleep(1)
    if not message_queue.empty():
        if message_queue.get() == "complate_msg":
            break

