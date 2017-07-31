#-*-coding:utf8-*-
__author__ = '何斌'
__datetime__='2017-7-15'
'''
pick the images only contains the face

'''

import os
import cv2
import time
from read_img import readAllImg

sourcePath = r"images/hb"
objectPath = r"datasets/hb"

#从源路径中读取所有图片放入一个list，然后逐一进行检查，把其中的脸扣下来，存储到目标路径中
def readPicSaveFace(sourcePath,objectPath,*suffix):
    if not os.path.exists(objectPath):
            os.mkdir(objectPath)
    try:
        #读取照片,注意第一个元素是文件名
        resultArray,_ = readAllImg(sourcePath,*suffix)

        #对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
        count = 1
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        for img in resultArray:
            # if type(img) != str:
            #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img 
            faces = face_cascade.detectMultiScale(gray,1.1,5)
            for (x, y, w, h) in faces:
                listStr = [str(int(time.time())), str(count)]  #以时间戳和读取的排序作为文件名称
                fileName = ''.join(listStr)
                f = cv2.resize(gray[y:(y + h), x:(x + w)], (128, 128))
                cv2.imwrite(objectPath+os.sep+'%s.jpg' % fileName, f)
                count += 1


    except IOError:
        print ("Error")

    else:
        print ('Already read '+str(count-1)+' Faces to Destination '+objectPath)

if __name__ == '__main__':
    readPicSaveFace(sourcePath,objectPath,'.jpg')










