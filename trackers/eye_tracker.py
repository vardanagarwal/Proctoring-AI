# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 19:21:18 2020

@author: hp
"""

import os
import time

import cv2
from dotenv import load_dotenv
from loguru import logger
import numpy as np

from trackers.face_detector import get_face_detector, find_faces
from trackers.face_landmarks import get_landmark_model, detect_marks

load_dotenv()


def eye_on_mask(mask, side, shape):
    """
    Create ROI on mask of the size of eyes and also find the extreme points of each eye

    Parameters
    ----------
    mask : np.uint8
        Blank mask to draw eyes on
    side : list of int
        the facial landmark numbers of eyes
    shape : Array of uint32
        Facial landmarks

    Returns
    -------
    mask : np.uint8
        Mask with region of interest drawn
    [l, t, r, b] : list
        left, top, right, and bottommost points of ROI

    """
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
    else:
        return 0

    
def contouring(thresh, mid, img, end_points, right=False):
    """
    Find the largest contour on an image divided by a midpoint and subsequently the eye position

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image of one side containing the eyeball
    mid : int
        The mid point between the eyes
    img : Array of uint8
        Original Image
    end_points : list
        List containing the exteme points of eye
    right : boolean, optional
        Whether calculating for right eye or left eye. The default is False.

    Returns
    -------
    pos: int
        the position where eyeball is:
            0 for normal
            1 for left
            2 for right
            3 for up

    """
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass
    
def process_thresh(thresh):
    """
    Preprocessing the thresholded image

    Parameters
    ----------
    thresh : Array of uint8
        Thresholded image to preprocess

    Returns
    -------
    thresh : Array of uint8
        Processed thresholded image

    """
    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):
    """
    Print the side where eye is looking and display on image

    Parameters
    ----------
    img : Array of uint8
        Image to display on
    left : int
        Position obtained of left eye.
    right : int
        Position obtained of right eye.

    Returns
    -------
    None.

    """
    if left == right and left != 0:
        text = ''
        if left == 1:
            print('Looking left')
            text = 'left'
        elif left == 2:
            print('Looking right')
            text = 'right'
        elif left == 3:
            print('Looking up')
            text = 'up'
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, text, (30, 30), font,  
                   1, (0, 255, 255), 2, cv2.LINE_AA)
        return text

face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cv2.namedWindow("image")
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass

cv2.createTrackbar("threshold", "image", 75, 255, nothing)

def track_eye(video_path, res_dict):
    eye_left_count = 0
    eye_right_count = 0
    gaze_direction = 0 # 1: left, 2: right
    sustained_gaze = False
    skip_count = int(os.getenv("FRAMETOANALYSE", 90))

    try:
        start_time = time.time()
        logger.info("Starting eye tracking")

        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        thresh = img.copy()
        frame_count = 0

        while True:
            ret, img = cap.read()

            if not ret:
                break
            
            rects = find_faces(img, face_model)
            thresh = img.copy()

            frame_count += 1

            if not frame_count % skip_count == 0:
                continue

            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask, end_points_left = eye_on_mask(mask, left, shape)
                mask, end_points_right = eye_on_mask(mask, right, shape)
                mask = cv2.dilate(mask, kernel, 5)
                
                eyes = cv2.bitwise_and(img, img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]
                mid = int((shape[42][0] + shape[39][0]) // 2)
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                threshold = cv2.getTrackbarPos('threshold', 'image')
                _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
                thresh = process_thresh(thresh)
                
                eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
                eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
                pos = print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)

                if pos == "left":
                    if gaze_direction != 1:
                        sustained_gaze = False
                    gaze_direction = 1
                    if not sustained_gaze:
                        eye_left_count += 1
                        sustained_gaze = True
                elif pos == "right":
                    if gaze_direction != 2:
                        sustained_gaze = False
                    gaze_direction = 2
                    if not sustained_gaze:
                        eye_right_count += 1
                        sustained_gaze = True
                else:
                    gaze_direction = 0
                    sustained_gaze = False

                # for (x, y) in shape[36:48]:
                #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
                
            # cv2.imshow('eyes', img)
            # cv2.imshow("image", thresh)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except Exception as e:
        logger.info(f"Error in eye_tracker: {e}")
        
    cap.release()
    # cv2.destroyAllWindows()

    logger.info(f"track_eye: {time.time() - start_time} secs")
    logger.info("Eye tracking completed")

    res_dict["Eye Left"] = eye_left_count
    res_dict["Eye Right"] = eye_right_count
    print(res_dict)
    return res_dict
