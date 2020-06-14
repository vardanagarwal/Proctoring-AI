# -*- coding: utf-8 -*-
"""
Created on Fri May  1 01:47:48 2020

@author: hp
"""

import tensorflow as tf
import cv2
import numpy as np
import time

interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
cap = cv2.VideoCapture(0)

while(True):
    for i in range(120):
        start = time.time()
        ret, image = cap.read()
        if ret==True:
            # resize to max dimension of images from training dataset
            image = cv2.resize(image, (300, 300), cv2.INTER_AREA)
            image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # apply normalization for trained dataset images
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image_RGB, 0))
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            labels = np.argmax(output_data.squeeze(), -1)
            # sh = labels.shape
            # labels = labels.flatten()
            # labels = np.array([255 if i == 15 else 0 for i in labels])
            # labels = labels.reshape(sh)
            # labels = labels.astype(np.uint8)
            cv2.imshow('Original', image)
            # cv2.imshow('Prediction', labels)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    end = time.time()
    fps = 120/(start-end)
    break

print(fps)
cap.release()
cv2.destroyAllWindows()




# while(True):
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (257, 257), cv2.INTER_AREA)
#     interpreter.set_tensor(input_details[0]['index'], frame)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     my_preds = my_preds.flatten()
#     my_preds = np.array([255 if i >= 0.5 else 0 for i in my_preds])
#     my_preds = my_preds.reshape(353, 353)
#     my_preds = my_preds.astype(np.uint8)
#     cv2.imshow('Original', frame)
#     cv2.imshow('Prediction', my_preds)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()

