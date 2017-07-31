#-*-coding:utf8-*-
__author__ = '何斌'
__datetime__='2017-7-15'

from read_data import read_name_list,read_file
from read_img import readAllImg
from train_model import Model
import cv2

def test_onePicture(path):
    model= Model(read_save=True)
    model.load()
    img = cv2.imread(path)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,5)
    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:(y + h), x:(x + w)], (128, 128))
    picType,prob = model.predict(face) 
    print (picType)
    name_list = read_name_list('G:\Python/FaceRecognition\datasets')
    if picType != -1:
    # if picType>=0 and picType<len(name_list):     
        print (name_list[picType],prob)
    else:
        print (" Don't know this person")

#读取文件夹下子文件夹中所有图片进行识别
def test_onBatch(path):
    model= Model(read_save=True)
    model.load()
    index = 0
    img_list,img_name_list = readAllImg(path,'.jpg')
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    for img in img_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,5)
        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:(y + h), x:(x + w)], (128, 128))
        picType,prob = model.predict(face)
        name_list = read_name_list('G:\Python/FaceRecognition\datasets')
        if picType != -1:
        # if picType>=0 and picType<len(name_list):
            print (img_name_list[index],":",name_list[picType],",",prob)
            index += 1
        else:
            print (" Don't know this person")

    return index

if __name__ == '__main__':
    test_onePicture(r"G:/Python/FaceRecognition/images/yangyang/1.jpg")
    # test_onBatch("G:/Python/faceRecognition-master/images/mix_pic")
  



