"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import cv2
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import scipy.misc
from multiprocessing import Process, Queue
import glob
import time
import dlib


image_extension = ".png"

model = "/root/src/facenet/src/models/20180402-114759"
videoName = "rtsp://admin:ABC_123456@172.17.208.150:554/Streaming/Channels/101?transportmode=unicast"
window_name = "人脸识别"
compare_dir = "/root/src/test/facenet/contributed/images/images_data_160"
#compare_dir = "/root/src/data"

detector = dlib.get_frontal_face_detector()


src_path,_ = os.path.split(os.path.realpath(__file__))
print(src_path)
temporary_dir = os.path.join(src_path,"data")
print(temporary_dir)

InfoQueue = Queue()
MessageQueue = Queue()
FrameQueue = Queue()


def main():
    processes = []

    p = Process(target=PersonCompare, args=(InfoQueue,MessageQueue,))
    p.start()
    processes.append(p)

    p = Process(target=FaceDectection, args=(FrameQueue,InfoQueue,MessageQueue,))
    p.start()
    processes.append(p)

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
        if n == 100:
            if process_flag == True:
                FrameQueue.put(frame)
            n = 1
            
        else:
            frameresize = cv2.resize(frame,(1280,800))
            cv2.imshow(window_name, frameresize)

        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            MessageQueue.put("complate_msg")
            MessageQueue.put("complate_msg")
            break        
        if c & 0xFF == ord('r'):
            process_flag = True

        if c & 0xFF == ord('s'):
            process_flag = False


 
    #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows() 

    for p in processes:
        p.join()
    


add = 0
def FaceDectection(frame_q,info_q,msg_q):
    index = 0
    num = 0

    if  not os.path.exists(temporary_dir):
        os.makedirs(temporary_dir)
    while 1:
        img = frame_q.get()
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

            if face.shape[0] > 300 :  # check picture size
                g_img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                lm = cv2.Laplacian(g_img, cv2.CV_64F).var()
                print("## lm : %d \n" % lm)
                if lm < 300:
                    continue
                
                num += 1
                if num >= 2:
                    file = os.path.join(temporary_dir,str(index)+image_extension)
                    image = cv2.resize(face,(160,160))
                    cv2.imwrite(file, image)
                    info_q.put(file)
                    num = 0
                    index += 1
                    if index > 1000:
                        index = 0



        if not msg_q.empty():
            if msg_q.get() == "complate_msg":
                break


def PersonCompare(info_queue,msg_q):
    image_size = (160,160)
    with tf.Graph().as_default():
      
        with tf.Session() as sess:

            #image_dir1 = os.path.expanduser(args.first_dir)
            #image_dir2 = os.path.expanduser(args.second_dir)
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            #image_size = (image_size, image_size)

            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            print('=== model: ')
            print(model)
            print('===input_map: ')
            print(input_map)
            facenet.load_model(model, input_map=input_map)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            embedding_size = embeddings.get_shape()[1]

            print('facenet embedding模型建立完毕')

            image_num = 0
            pass_num = 0

            while 1:
                input_img = info_queue.get()

                #print(input_img)
                #print("compare dir: %s" % compare_dir)

                for parent, dirnames, filenames in os.walk(compare_dir):
                    #print(dirnames)
                    for username in dirnames:
                        userdir = os.path.join(compare_dir,username)

                        img_files = os.path.join(userdir,"*" + image_extension)
                
                        img_name_list = glob.glob(img_files)

                        for compare_img in img_name_list:
                            scaled_reshape = []
                            image1 = scipy.misc.imread(input_img, mode='RGB')
                            image1 = cv2.resize(image1, image_size,  interpolation=cv2.INTER_CUBIC)
                            image1 = facenet.prewhiten(image1)
                            scaled_reshape.append(image1.reshape(-1,160,160,3))
                            emb_array1 = np.zeros((1, embedding_size))
                            emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False })[0]

                            image2 = scipy.misc.imread(compare_img, mode='RGB')
                            image2 = cv2.resize(image2, image_size, interpolation=cv2.INTER_CUBIC)
                            image2 = facenet.prewhiten(image2)
                            scaled_reshape.append(image2.reshape(-1,160,160,3))
                            emb_array2 = np.zeros((1, embedding_size))
                            emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False })[0]

                            dist = np.sqrt(np.sum(np.square(emb_array1[0]-emb_array2[0])))
                            print("%s：%f "% (username,dist))
                            if dist < 1.0:
                                print("pass num += 1")
                                pass_num += 1

                            image_num += 1
                                #print(" 第%d组照片是同一个人 "%num)
                        if image_num == 0:
                            pass_num = 0
                            continue
                        pass_rate = pass_num/image_num
                        print("%s 通过率: %f " % (username,pass_rate))
                        pass_num = 0
                        image_num = 0
                        if pass_rate > 0.9 :
                            print("=========== 你是: %s\n" % username)
                            #break
                    break


                if not msg_q.empty():
                    if msg_q.get() == "complate_msg":
                        break

"""
              
def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, distance_metric, subtract_mean, use_flipped_images, use_fixed_image_standardization):
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
    
"""

if __name__ == '__main__':
    main()
