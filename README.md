# Proctoring-AI
Project to create an AI for online proctoring. It has three vision based functionalities:
1. Track eyeballs and report if candidate is looking left, right or up.
2. Instance segmentation to count number of people and report if no one or more than one person detected.
3. Record the distance between lips at starting. Report if candidate opens his mouth.

Might add more if get any good ideas.

For easier understanding of each functionality see the codes in individual_code where each functionallity is implemented individually. Functionality 1 is dlib_eyes.py, functionality 2 is yolov3.py and functionality 3 is dlib_mouth.py. A tutorial of eye tracking of dlib_eyes.py can be found here https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6.

A combined version using multiprocessing coming soon.
