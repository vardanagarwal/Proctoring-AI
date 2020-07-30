# Proctoring-AI

Project to create an automated proctoring system where the user can be monitored automatically through the webcam and microphone. The project is divided into two parts: vision and audio based functionalities. An in-depth explanation of the project can be found on my [medium article](https://towardsdatascience.com/automating-online-proctoring-using-ai-e429086743c8?source=friends_link&sk=fbc385d1a8c55628a916dc714747f276)

### Prerequisites

For vision:
```
Tensorflow
OpenCV
Dlib
```
For audio:
```
pyaudio
speech_recognition
nltk
```

## Vision

It has four vision based functionalities:
1. Track eyeballs and report if candidate is looking left, right or up.
2. Instance segmentation to count number of people and report if no one or more than one person detected.
3. Record the distance between lips at starting. Report if candidate opens his mouth.
4. Find and report any instances of mobile phones.
5. Head pose estimation to find where the person is looking.

### Run
Run `main.py`. To record mouth distances press r as indicated in video displayed. Then using multithreading, seperate threads are created for tracking eyes and mouth and one for counting people. To quit press q twice. This is for the first four functionalities.
For head pose detection run [head_pose_detector](../../blob/old_master/LICENSE.md)

#### Tutorials and Understanding

For easier understanding of each functionality see the codes in [individual_code](../../blob/old_master/individual_codes) where each functionallity is implemented individually. 
1. `dlib_eyes.py` is to track eyes. [Article](https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6?source=friends_link&sk=d9db46e2f41258c6c23d18792775d2a5) for more details.
2. `yolov3.py` is for counting persons and detecting mobile phones. [Article](https://medium.com/analytics-vidhya/count-people-in-webcam-using-yolov3-tensorflow-f407679967d5?source=friends_link&sk=95ae7a010eeef429a407a7a2de2ff8ec) for more details.
3. `dlib_mouth.py` is for checking if mouth is open or close.

`head_pose_estimation.py` is used for finding where the head is facing. [Article](https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a?source=friends_link&sk=0bae01db2759930197bfd33777c9eaf4)

## Audio
It is divided into two parts:
1. Audio from the microphone is recording and converted to text using Google's speech recognition API. A different thread is used to call the API such that the recording portion is not disturbed a lot, which processes the last one, appends its data to a text file and deletes it.
2. NLTK we remove the stopwods from that file. The question paper (in txt format) is taken whose stopwords are also removed and their contents are compared. Finally, the common words along with its number are presented to the proctor.


### Run

Run `audio_part.py`

## Additional Work

Dlib's frontal face HOG detector is used to find faces. However, it does not give very good results. In [face_detection](../../blob/old_master/face_detection) different face detection models are compared and OpenCV's DNN module provides best result and the results are present in [this article](https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c?source=friends_link&sk=c9e2807cf216115d7bb5a9b827bb26f8).

### To do
Replace the HOG based descriptor by OpenCV's DNN modules Caffe model and it will also solve the issues created by side faces and occlusion.
Replace the dlib based facial landmarks with the CNN based facial landmarks as used in head_pose_detector.
It will make this a lot hotch potch so I will retire this branch and make a new one with even more functionalites, better documentation and remove multithreading in the vision part.

## Contributing

If you have any other ideas make a pull request. Consider updating the README as well.

### Problems
The YOLOv3 has a very less fps like around 0.9 on a CPU.
Speech to text conversion which might not work well for all dialects.

## License

This project is licensed under the MIT License - see the [LICENSE.md](../../blob/old_master/LICENSE.md) file for details
