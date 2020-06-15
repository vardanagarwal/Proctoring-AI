# -*- coding: utf-8 -*-
"""
Created on Fri May  1 04:59:29 2020

@author: hp
"""

import numpy as np
import cv2
from model import Deeplabv3

# Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# as original image.  Normalization matches MobileNetV2

trained_image_width=512 
mean_subtraction_value=127.5

cap = cv2.VideoCapture(0)

while(True):
    ret, image = cap.read()
    if ret==True:
        # resize to max dimension of images from training dataset
        image = cv2.resize(image, (512, 512), cv2.INTER_AREA)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # apply normalization for trained dataset images
        normalized_image = (image_RGB / mean_subtraction_value) - 1.
        
        # make prediction
        deeplab_model = Deeplabv3()
        res = deeplab_model.predict(np.expand_dims(normalized_image, 0))
        labels = np.argmax(res.squeeze(), -1)
        
        sh = labels.shape
        labels = labels.flatten()
        labels = np.array([255 if i == 15 else 0 for i in labels])
        labels = labels.reshape(sh)
        labels = labels.astype(np.uint8)
        cv2.imshow('Original', image)
        cv2.imshow('Prediction', labels)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()


