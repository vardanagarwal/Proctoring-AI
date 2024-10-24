# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 01:04:44 2020

@author: hp
"""

import os
import time

import cv2
from dotenv import load_dotenv
from loguru import logger

from trackers.face_detector import get_face_detector, find_faces
from trackers.face_landmarks import get_landmark_model, detect_marks, draw_marks


load_dotenv()


face_model = get_face_detector()
landmark_model = get_landmark_model()
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3
font = cv2.FONT_HERSHEY_SIMPLEX 


def mouth_opening_detector(video_path, res_dict):
    start = time.time()
    logger.info("Starting mouth opening detection")
    frames_recorded = 0
    frame_count = 0
    skip_count = int(os.getenv("FRAMETOANALYSE", 90))
    try:
        cap = cv2.VideoCapture(video_path)

        mouth_open_detected = 0
        sustained_detection = False

        while True:
            ret, img = cap.read()

            if not ret:
                break

            rects = find_faces(img, face_model)
            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                # draw_marks(img, shape)
                # cv2.putText(img, 'Recording Mouth distances', (30, 30), font,
                #             1, (0, 255, 255), 2)
                # cv2.imshow("Output", img)
            
            try:
                if frames_recorded < 50:
                    for i, (p1, p2) in enumerate(outer_points):
                        d_outer[i] += shape[p2][1] - shape[p1][1]
                    for i, (p1, p2) in enumerate(inner_points):
                        d_inner[i] += shape[p2][1] - shape[p1][1]
                    frames_recorded += 1
                else:
                    break
            except Exception as e:
                logger.error(e)
                continue

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     break

        #     for rect in rects:
        #         shape = detect_marks(img, landmark_model, rect)
        #         draw_marks(img, shape)
        #         cv2.putText(img, 'Press r to record Mouth distances', (30, 30), font,
        #                     1, (0, 255, 255), 2)
        #         cv2.imshow("Output", img)
        #     if cv2.waitKey(1) & 0xFF == ord('r'):
        #         for i in range(100):
        #             for i, (p1, p2) in enumerate(outer_points):
        #                 d_outer[i] += shape[p2][1] - shape[p1][1]
        #             for i, (p1, p2) in enumerate(inner_points):
        #                 d_inner[i] += shape[p2][1] - shape[p1][1]
        #         break
        # cv2.destroyAllWindows()

        d_outer[:] = [x / 100 for x in d_outer]
        d_inner[:] = [x / 100 for x in d_inner]

        while True:
            ret, img = cap.read()
            if not ret:
                break
            
            rects = find_faces(img, face_model)

            frame_count += 1
            if not frame_count % skip_count == 0:
                continue

            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                cnt_outer = 0
                cnt_inner = 0
                # draw_marks(img, shape[48:])
                for i, (p1, p2) in enumerate(outer_points):
                    if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                        cnt_outer += 1 
                for i, (p1, p2) in enumerate(inner_points):
                    if d_inner[i] + 2 <  shape[p2][1] - shape[p1][1]:
                        cnt_inner += 1
                
                if cnt_outer > 3 and cnt_inner > 2:
                    if not sustained_detection:
                        logger.info('Mouth open')
                        mouth_open_detected += 1
                        sustained_detection = True
                    # cv2.putText(img, 'Mouth open', (30, 30), font,
                    #             1, (0, 255, 255), 2)
                else:
                    sustained_detection = False
                
                # show the output image with the face detections + facial landmarks

            # cv2.imshow("Output", img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    except Exception as e:
        logger.info(f"Error in mouth_opening_detector: {e}")
        
    cap.release()
    # cv2.destroyAllWindows()
    logger.info(f"mouth_opening_detector : {time.time() - start} secs")
    res_dict["Mouth Open"] = mouth_open_detected
    return res_dict
