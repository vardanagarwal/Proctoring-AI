# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 03:40:59 2020

@author: hp
"""
import cv2
import dlib
from mtcnn.mtcnn import MTCNN
import numpy as np

detector1 = MTCNN()
detector2 = dlib.get_frontal_face_detector()
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
classifier2 = cv2.CascadeClassifier('models/haarcascade_frontalface2.xml')

cap = cv2.VideoCapture('video/occlusion.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX 
while(True):
    ret, img = cap.read()
    if ret == True:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        height, width = img.shape[:2]
        img1 = img.copy()
        img2 = img.copy()
        img3 = img.copy()
        # detect faces in the image
        faces1 = detector1.detect_faces(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces2 = detector2(gray, 1)
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                        1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces3 = net.forward()
        faces4 = classifier2.detectMultiScale(img)
        
        # display faces on the original image
        for result in faces1:
            x, y, w, h = result['box']
            x1, y1 = x + w, y + h
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img, 'mtcnn', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        for result in faces2:
            x = result.left()
            y = result.top()
            x1 = result.right()
            y1 = result.bottom()
            cv2.rectangle(img1, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img1, 'dlib', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        for i in range(faces3.shape[2]):
            confidence = faces3[0, 0, i, 2]
            if confidence > 0.5:
                box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img2, 'dnn', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
                
        for result in faces4:
            x, y, w, h = result
            x1, y1 = x + w, y + h
            cv2.rectangle(img3, (x, y), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img3, 'haar', (30, 30), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        h1 = cv2.hconcat([img3, img1])
        h2 = cv2.hconcat([img, img2])
        fin = cv2.vconcat([h1, h2])

        cv2.imshow("mtcnn", img)
        cv2.imshow("dlib", img1)
        cv2.imshow("dnn", img2)
        cv2.imshow("haar", img3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


           
cap.release()
cv2.destroyAllWindows()
