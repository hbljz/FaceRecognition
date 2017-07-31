import cv2
import numpy as np
cv2.namedWindow("test")#命名一个窗口
cap=cv2.VideoCapture(0)#打开1号摄像头
success, frame = cap.read()#读取一桢图像，前一个返回值是是否成功，后一个返回值是图像本身
color = (0,0,0)#设置人脸框的颜色
classfier=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')#定义分类器
while success:
    success, frame = cap.read()
    size=frame.shape[:2]#获得当前桢彩色图像的大小
    image=np.zeros(size,dtype=np.float16)#定义一个与当前桢图像大小相同的的灰度图像矩阵
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#将当前桢图像转换成灰度图像（这里有修改）
    cv2.equalizeHist(image, image)#灰度图像进行直方图等距化
    #如下三行是设定最小图像的大小
    divisor=8
    h, w = size
    minSize=(round(w/divisor), round(h/divisor))#这里加了一个取整函数
    faceRects = classfier.detectMultiScale(image, 1.2, 5,minSize=(5,5))#人脸检测
    if len(faceRects)>0:#如果人脸数组长度大于0
        for faceRect in faceRects: #对每一个人脸画矩形框
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x, y), (x+w, y+h), color)
    cv2.imshow("test", frame)#显示图像
    key=cv2.waitKey(10)
    c = chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break
cv2.destroyWindow("test")