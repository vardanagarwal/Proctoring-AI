# Proctoring-AI

Project to create an automated proctoring system where the user can be monitored automatically through the webcam and microphone. The project is divided into two parts: vision and audio based functionalities. An explanation of some functionalities of the project can be found on my [medium article](https://towardsdatascience.com/automating-online-proctoring-using-ai-e429086743c8?source=friends_link&sk=fbc385d1a8c55628a916dc714747f276)

### Prerequisites

For vision:
```
Tensorflow
OpenCV
sklearn=0.19.1 # the model used was trained with this version and does not support recent ones.
```
For audio:
```
pyaudio
speech_recognition
nltk
```

## Vision

It has six vision based functionalities right now:
1. Track eyeballs and report if candidate is looking left, right or up.
2. Find if the candidate opens his mouth by recording the distance between lips at starting.
3. Instance segmentation to count number of people and report if no one or more than one person detected.
4. Find and report any instances of mobile phones.
5. Head pose estimation to find where the person is looking.
6. Face spoofing detection

### Face detection
Earlier, Dlib's frontal face HOG detector was used to find faces. However, it did not give very good results. In [face_detection](../../tree/master/face_detection) different face detection models are compared and OpenCV's DNN module provides best result and the results are present in [this article](https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c?source=friends_link&sk=c9e2807cf216115d7bb5a9b827bb26f8).

It is implemented in `face_detector.py` and is used for tracking eyes, mouth opening detection, head pose estimation, and face spoofing.

### Facial Landmarks
Earlier, Dlib's facial landmarks model was used but it did not give good results when face was at an angle. Now, a model provided in this [repository](https://github.com/yinguobing/cnn-facial-landmark) is used. An article comparing them will be written soon.

It is implemented in `face_landmarks.py` and is used for tracking eyes, mouth opening detection, and head pose estimation.

#### Note
If you want to use dlib models then checkout the [old-master branch](https://github.com/vardanagarwal/Proctoring-AI/tree/old_master).

### Eye tracking
`eye_tracker.py` is to track eyes. A detailed explanation is provided in this [article](https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6?source=friends_link&sk=d9db46e2f41258c6c23d18792775d2a5). However, it was written using dlib.

### Mouth Opening Detection
`mouth_opening_detector.py` is used to check if the candidate opens his/her mouth during the exam after recording it initially. It's explanation can be found in the main article, however, it is using dlib which can be easily changed to the new models.

### Person counting and mobile phone detection
`person_and_phone.py` is for counting persons and detecting mobile phones. YOLOv3 is used in Tensorflow 2 and it is explained in this [article](https://medium.com/analytics-vidhya/count-people-in-webcam-using-yolov3-tensorflow-f407679967d5?source=friends_link&sk=95ae7a010eeef429a407a7a2de2ff8ec) for more details.

### Head pose estimation
`head_pose_estimation.py` is used for finding where the head is facing. An explanation is provided in this [article](https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a?source=friends_link&sk=0bae01db2759930197bfd33777c9eaf4)

### Face spoofing
`face_spoofing.py` is used for finding whether the face is real or a photograph or image. An explanation is provided in this [article](https://medium.com/visionwizard/face-spoofing-detection-in-python-e46761fe5947).

## Audio
It is divided into two parts:
1. Audio from the microphone is recording and converted to text using Google's speech recognition API. A different thread is used to call the API such that the recording portion is not disturbed a lot, which processes the last one, appends its data to a text file and deletes it.
2. NLTK we remove the stopwods from that file. The question paper (in txt format) is taken whose stopwords are also removed and their contents are compared. Finally, the common words along with its number are presented to the proctor.

The code for this part is available in `audio_part.py`

## To do
1. ~~Replace the HOG based descriptor by OpenCV's DNN modules Caffe model and it will also solve the issues created by side faces and occlusion.~~
2. ~~Replace the dlib based facial landmarks with the CNN based facial landmarks as used in head_pose_detector.~~
3. Make a better face spoofing model as the accuracy is not good currently.
4. Use a smaller and faster model inplace of YOLOv3 that can give good FPS on a CPU.
5. Add a vision based functionality: face recognition such that no one else replaces the candidate and gives the exam midway.
6. Add a vision based functionality: id-card verification.
7. Update README with videos of each functionality and the FPS obtained.
8. ~~Add documentation (docstring) in functions in codes.~~

### Problems
Speech to text conversion which might not work well for all dialects.

## Contributing

If you have any other ideas make a pull request. Consider updating the README as well.

## License

This project is licensed under the MIT License - see the [LICENSE.md](../../tree/master/LICENSE.md) file for details
