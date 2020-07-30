# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:52:00 2020

@author: hp
"""

import cv2
import numpy as np

def get_face_detector(modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel",
                      configFile = "models/deploy.prototxt"):
    """
    Get the face detection caffe model of OpenCV's DNN module
    
    Parameters
    ----------
    modelFile : string, optional
        Path to model file. The default is "models/res10_300x300_ssd_iter_140000.caffemodel".
    configFile : string, optional
        Path to config file. The default is "models/deploy.prototxt".

    Returns
    -------
    model : dnn_Net

    """
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt"
    model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def find_faces(img, model):
    """
    Find the faces in an image
    
    Parameters
    ----------
    img : np.uint8
        Image to find faces from
    model : dnn_Net
        Face detection model

    Returns
    -------
    faces : list
        List of coordinates of the faces detected in the image

    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces

def draw_faces(img, faces):
    """
    Draw faces on image

    Parameters
    ----------
    img : np.uint8
        Image to draw faces on
    faces : List of face coordinates
        Coordinates of faces to draw

    Returns
    -------
    None.

    """
    for x, y, x1, y1 in faces:
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 3)
        