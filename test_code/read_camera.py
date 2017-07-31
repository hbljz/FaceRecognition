# -*- coding:utf-8 -*-
__author__ = '何斌'
__datetime__='2017-7-15'

import cv2,time
from train_model import Model
from read_data import read_name_list

dataset_path='G:\Python/FaceRecognition\datasets'

class Camera_reader(object):
    #在初始化camera的时候建立模型，并加载已经训练好的模型
    def __init__(self):
        self.model = Model(read_save=True)
        self.model.load()
        self.img_size = 128
        #opencv文件中人脸级联文件的位置，用于帮助识别图像或者视频流中的人脸
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        #读取dataset数据集下的子文件夹名称
        self.name_list = read_name_list(dataset_path)

    def build_camera(self):
        #打开摄像头并开始读取画面
        # cv2.namedWindow("Face_Recognization")
        cameraCapture = cv2.VideoCapture(0)
        while cameraCapture.isOpened():
             success, frame = cameraCapture.read()
             if success==True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #图像灰化
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5) #识别人脸
                for (x, y, w, h) in faces:
                    face_img = gray[x:x + w, y:y + h]
                    face_img = cv2.resize(face_img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                    label,prob = self.model.predict(face_img)  #利用模型对cv2识别出的人脸进行比对
                    if prob >0.7:#如果模型认为概率高于70%则显示为模型中已有的label
                        show_name = self.name_list[label]
                    else:
                        show_name = 'Stranger'
                    cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2)  #显示名字
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)  #在人脸区域画一个正方形出来
                cv2.imshow("Face_Recognization", frame)
                key=cv2.waitKey(10)
                c = chr(key & 255)
                if c in ['q','Q', chr(27)]:         #chr(27)是Esc键
                    break
        cameraCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = Camera_reader()
    camera.build_camera()


