import cv2  
import time  
  
if __name__ == '__main__':  
  
    # cv.NamedWindow("camera",1)  
    # capture = cv.CaptureFromCAM(0)            #开启摄像头  
    # capture = cv.CaptureFromFile("Video.avi")   #打开一个视频文件  
   	cameraCapture = cv2.VideoCapture(0)
   	num = 0
   	save_path=r"G:/Python/FaceRecognition/images/hb/"
   	while cameraCapture.isOpened():
   		success, img = cameraCapture.read()
   		if success==True:
   			cv2.imshow("camera",img)
   			key=cv2.waitKey(10)
   			c = chr(key & 255)
   			if c in ['q','Q', chr(27)]:         #chr(27)是Esc键
   				break
   			time.sleep(2)
   			num = num+1
   			filename = "frmaes_%s.jpg" % num
   			cv2.imwrite(save_path+'%s' % filename,img)
   	cameraCapture.release()
   	cv2.destroyAllWindows()