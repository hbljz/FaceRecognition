# coding:utf-8
#__author__ = 'hebin'
#__date__ = '2017/7/28'
#__Desc__ = 人脸检测小例子
   
import sys  
import cv2
# 待检测的图片路径
imagepath = r'G:/Python/FaceRecognition/images/yihuan/3.jpg'

# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

# 读取图片
image = cv2.imread(imagepath)

#图片灰度化，减少计算量
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# 探测图片中的人脸
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (5,5),
    )

print ("发现{0}个人脸!".format(len(faces)))

for(x,y,w,h) in faces:
    # cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
    cv2.circle(image,(int((x+x+w)/2),int((y+y+h)/2)),int(w/2),(0,255,0),2)

cv2.imshow("Find Faces!",image)
cv2.waitKey(0)