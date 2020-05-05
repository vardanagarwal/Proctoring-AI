# -*- coding: utf-8 -*-
"""
Created on Wed May  6 03:56:41 2020

@author: hp
"""
import cv2
from dlib_helper import shape_to_np

def return_distances(detector, predictor):
    outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
    d_outer = [0]*5
    inner_points = [[61, 67], [62, 66], [63, 65]]
    d_inner = [0]*3
    cap = cv2.VideoCapture(0)
    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for rect in rects:
    
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        # show the output image with the face detections + facial landmarks
        cv2.putText(img, 'Press r to record mouth distance', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            for i in range(100):
                for i, (p1, p2) in enumerate(outer_points):
                    d_outer[i] += shape[p2][1] - shape[p1][1]
                for i, (p1, p2) in enumerate(inner_points):
                    d_inner[i] += shape[p2][1] - shape[p1][1]
            break
    cv2.destroyAllWindows()
    d_outer[:] = [x / 100 for x in d_outer]
    d_inner[:] = [x / 100 for x in d_inner]
    return d_outer, d_inner