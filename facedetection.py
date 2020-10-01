# -*- coding: utf-8 -*-
'''

'''

import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame):
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 5)
        smile = smileCascade.detectMultiScale(roi_gray, 1.1, 10)
        
        for (x2,y2,w2,h2) in eyes:
            cv2.rectangle(roi_frame,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
            
        for (x3,y3,w3,h3) in smile:
            cv2.rectangle(roi_frame,(x3,y3),(x3+w3,y3+h3), (0,0,255),4)
        
    return frame
        
        
videoCapture =  cv2.VideoCapture(0)
while True:
    _, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()