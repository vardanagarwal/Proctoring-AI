# Proctoring-AI

Project to create an automated proctoring system where the user can be monitored automatically through the webcam and microphone. The project is divided into two parts: vision and audio based functionalities. An explanation of some functionalities of the project can be found on my [medium article](https://towardsdatascience.com/automating-online-proctoring-using-ai-e429086743c8?source=friends_link&sk=fbc385d1a8c55628a916dc714747f276).

### Prerequisites
To run the programs in this repo, do the following:
- create a virtual environment using the command:
  - `python -m venv venv`
- activate the virtual environment
  - `./venv/Scripts/activate` (windows users)
  - `source ./venv/bin/activate` (mac and linux users)
- install the requirements
  - `pip install --upgrade pip` (to upgrade pip)
  - `pip install -r requirements.txt`

Once the requirements have been installed, The programs will run successfully.
Except for the `person_and_phone.py` script which requires a model to be downloaded.

More on that later.


For vision:
```
Tensorflow>2
OpenCV
sklearn=0.19.1 # for face spoofing. 
The model used was trained with this version and does not support recent ones.
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

An additional quantized model is also added for face detector as described in [Issue 14](https://github.com/vardanagarwal/Proctoring-AI/issues/14). This can be used by setting the parameter `quantized` as True when calling the `get_face_detector()`. On quick testing of face detector on my laptop the normal version gave ~17.5 FPS while the quantized version gave ~19.5 FPS. This would be especially useful when deploying on edge devices due to it being uint8 quantized.

### Facial Landmarks
Earlier, Dlib's facial landmarks model was used but it did not give good results when face was at an angle. Now, a model provided in this [repository](https://github.com/yinguobing/cnn-facial-landmark) is used. A comparison between them and the reason for choosing the new Tensorflow based model is shown in this [article](https://towardsdatascience.com/robust-facial-landmarks-for-occluded-angled-faces-925e465cbf2e?source=friends_link&sk=505eb1101576227f4c38474092dd4c22).

It is implemented in `face_landmarks.py` and is used for tracking eyes, mouth opening detection, and head pose estimation.

#### Note
If you want to use dlib models then checkout the [old-master branch](https://github.com/vardanagarwal/Proctoring-AI/tree/old_master).

### Eye tracking
`eye_tracker.py` is to track eyes. A detailed explanation is provided in this [article](https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6?source=friends_link&sk=d9db46e2f41258c6c23d18792775d2a5). However, it was written using dlib.

![eye tracking](../../blob/master/gifs/1.gif)

### Mouth Opening Detection
`mouth_opening_detector.py` is used to check if the candidate opens his/her mouth during the exam after recording it initially. It's explanation can be found in the main article, however, it is using dlib which can be easily changed to the new models.

![Mouth opening detection](../../blob/master/gifs/2.gif)

### Person counting and mobile phone detection
`person_and_phone.py` is for counting persons and detecting mobile phones. YOLOv3 is used in Tensorflow 2 and it is explained in this [article](https://medium.com/analytics-vidhya/count-people-in-webcam-using-yolov3-tensorflow-f407679967d5?source=friends_link&sk=95ae7a010eeef429a407a7a2de2ff8ec) for more details.

![person counting and phone detection](../../blob/master/gifs/3.gif)

### Head pose estimation
`head_pose_estimation.py` is used for finding where the head is facing. An explanation is provided in this [article](https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a?source=friends_link&sk=0bae01db2759930197bfd33777c9eaf4)

![head pose estimation](../../blob/master/gifs/4.gif)

### Face spoofing
`face_spoofing.py` is used for finding whether the face is real or a photograph or image. An explanation is provided in this [article](https://medium.com/visionwizard/face-spoofing-detection-in-python-e46761fe5947). The model and working is taken from this Github [repo](https://github.com/ee09115/spoofing_detection).

![face spoofing](../../blob/master/gifs/5.gif)

### FPS obtained

Functionality | On Intel i5
--- | ---
Eye Tracking | 7.1
Mouth Detection | 7.2
Person and Phone Detection | 1.3
Head Pose Estimation | 8.5
Face Spoofing | 6.9

If you testing on a different processor a GPU consider making a pull request to add the FPS obtained on that processor.


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
7. ~~Update README with videos of each functionality and the FPS obtained.~~
8. ~~Add documentation (docstring) in functions in codes.~~

### Problems
Speech to text conversion which might not work well for all dialects.

## Contributing

If you have any other ideas or do any step of to do consider making a pull request . Please update the README as well in the pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](../../tree/master/LICENSE.md) file for details. However, the facial landmarks detection model is trained on non-commercial use datasets so I am not sure if that is allowed to be used for commercial purposes or not.

## Like what I am doing
<a href="https://www.buymeacoffee.com/vardan" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

