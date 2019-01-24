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
import tensorflow as tf

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

# 改变图片的亮度与对比度
def relight(img, light=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

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

def ConvertUseTensorflowMtcnn(in_queue,msg_q):
    margin = 32
    index_num = 0
    target_dir = os.path.join(img_dir,"images_data_%d" % target_img_size)
    output_user_dir = os.path.join(target_dir, name)
    if not os.path.exists(output_user_dir):
        os.makedirs(output_user_dir)

    minsize = 30 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor


    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    while 1:
        file_name = in_queue.get()
        print("====== file: %s" % file_name)
        img = cv2.imread(file_name)
        
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        print(bounding_boxes)
        nrof_faces = bounding_boxes.shape[0]
        print(nrof_faces)
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            print("====nrof_face: %d  img size:"  % nrof_faces)
            print(img_size) 
            if nrof_faces == 1:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                det_arr.append(det[index,:])
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                scaled = misc.imresize(cropped, (target_img_size, target_img_size), interp='bilinear')

                output_filename = os.path.join(output_user_dir, ("%d" % index_num) +'.png')

                filename_base, file_extension = os.path.splitext(output_filename)

                output_filename_n = "{}{}".format(filename_base, file_extension)

                misc.imsave(output_filename_n, scaled)

                index_num += 1
        #if os.path.exists(file_name):
            #os.remove(file_name)
        if index_num >= 5:
            msg_q.put("complate_msg")
            msg_q.put("complate_msg")
            msg_q.put("complate_msg")
            break



add = 30
def FaceDectection(in_queue,out_convert_queue,msg_q):
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
                out_convert_queue.put(nm)
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

p = Process(target=ConvertUseTensorflowMtcnn, args=(convert_queue,message_queue,))
p.start()
processes.append(p)


for p in processes:
    p.join()

while 1:
    time.sleep(1)
    if not message_queue.empty():
        if message_queue.get() == "complate_msg":
            break

