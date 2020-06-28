# Proctoring-AI
An in-depth explanation of the project can be found on my medium article: https://towardsdatascience.com/automating-online-proctoring-using-ai-e429086743c8?source=friends_link&sk=fbc385d1a8c55628a916dc714747f276

## Vision
Project to create an AI for online proctoring. It has four vision based functionalities:
1. Track eyeballs and report if candidate is looking left, right or up.
2. Instance segmentation to count number of people and report if no one or more than one person detected.
3. Record the distance between lips at starting. Report if candidate opens his mouth.
4. Find and report any instances of mobile phones

Might add more if get any good ideas.

Run main.py. To record mouth distances press r as indicated in video displayed. Then using multithreading, seperate threads are created for tracking eyes and mouth and one for counting people. To quit press q twice.

*Problems:* The YOLOv3 has a very less fps like around 0.9 on a CPU. So a GPU will be required as a hardware requirement if somebody wants to use this code in production. I tried using HOG but it did not give good accuracy. If anyone can help in this regard, he is more than welcome to.

*Note:* This system does not eliminate the use of a human proctor. This can only be used to aid them and can provide a option for a proctor to monitor more than candidate.

For easier understanding of each functionality see the codes in individual_code where each functionallity is implemented individually. 
1. dlib_eyes.py is to track eyes.
2. yolov3.py is for counting persons and detecting mobile phones.
3. dlib_mouth.py is for checking if mouth is open or close.

A tutorial of eye tracking of dlib_eyes.py can be found here https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6?source=friends_link&sk=d9db46e2f41258c6c23d18792775d2a5.

A tutorial of yolov3.py can be found here https://medium.com/analytics-vidhya/count-people-in-webcam-using-yolov3-tensorflow-f407679967d5?source=friends_link&sk=95ae7a010eeef429a407a7a2de2ff8ec.

Requirements - Tensorflow, OpenCV, and Dlib

## Audio
The idea is to record audio from the microphone and convert it to text using Google's speech recognition API. The API needed continous voice from the microphone which is not plausible so the audio is recorded in chunks such that a lot of space is not required (a ten second wav file had size of 1.5 mb so a three hour exam would have a lot). A different thread is used to call the API such that the recording portion is not disturbed a lot, which processes the last one, appends its data to a text file and deletes it.
After that using NLTK we remove the stopwords from it. The question paper (in txt format) is taken whose stopwords are also removed and their contents are compared. We assume if someone wants to cheat, they will speak something from the question paper. Finally, the common words along with its number are presented to the proctor. The proctor can also look at the text file which has all the words spoken.

The whole code of Audio part is present in audio_part.py.

Requirements - pyaudio, speech_recognition, and nltk.

*Note* - This part heavily relies on the speech to text conversion which might not work well for all dialects.
