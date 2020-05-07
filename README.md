# Proctoring-AI
Project to create an AI for online proctoring. It has three vision based functionalities:
1. Track eyeballs and report if candidate is looking left, right or up.
2. Instance segmentation to count number of people and report if no one or more than one person detected.
3. Record the distance between lips at starting. Report if candidate opens his mouth.

Might add more if get any good ideas.

Run main.py. To record mouth distances press r as indicated in video displayed. Then using multithreading seperate threads are created for tracking eyes and mouth and one for counting people. To quit press q twice.

*Problems:* The YOLOv3 has a very less fps like around 1(i'll calculate and update) on a CPU. So a GPU will be required as a hardware requirement if somebody wants to use this code in production. I tried using HOG but it did not give good accuracy. If anyone can help in this regard, he is more than welcome to.

*Note:* This system does not eliminate the use of a human proctor. This can only be used to aid them and can provide a option for a proctor to monitor more than candidate.

For easier understanding of each functionality see the codes in individual_code where each functionallity is implemented individually. Functionality 1 is dlib_eyes.py, functionality 2 is yolov3.py and functionality 3 is dlib_mouth.py. A tutorial of eye tracking of dlib_eyes.py can be found here https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6?source=friends_link&sk=d9db46e2f41258c6c23d18792775d2a5. A tutorial of yolov3.py can be found here https://medium.com/analytics-vidhya/count-people-in-webcam-using-yolov3-tensorflow-f407679967d5?source=friends_link&sk=95ae7a010eeef429a407a7a2de2ff8ec.

My partner is doing something speech based for this. That will be updated when he is done.
