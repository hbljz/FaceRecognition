import os,sys
import cv2
cv2.namedWindow("camera")
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
cameraCapture = cv2.VideoCapture(0)
while cameraCapture.isOpened():
    flag, frame = cameraCapture.read();    
    if flag == True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5,minSize = (5,5))
        for (x, y, w, h) in faces:
        	show_name = 'Stranger'
        	cv2.putText(frame, show_name,(x, y - 20),
        		cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0), 2)
        	cv2.rectangle(frame, (x, y),(x + w, y + h),(0, 0, 255), 2)
        	cv2.imshow("camera", frame)
        key=cv2.waitKey(10)
        c = chr(key & 255)
        if c in ['q','Q', chr(27)]:			#chr(27)是Esc键
        	break
cameraCapture.release()
cv2.destroyAllWindows()
# (frame.shape[1] / 10, frame.shape[0] / 10)