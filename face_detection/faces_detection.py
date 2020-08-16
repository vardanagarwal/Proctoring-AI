from mtcnn.mtcnn import MTCNN
import cv2
import dlib
import numpy as np
import os
		
detector1 = MTCNN()
detector2 = dlib.get_frontal_face_detector()
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

classifier2 = cv2.CascadeClassifier('models/haarcascade_frontalface2.xml')
images = os.listdir('faces')
# os.makedirs('faces/dlib')
# os.makedirs('faces/mtcnn')
# os.makedirs('faces/dnn')
# os.makedirs('faces/haar')

for image in images:
    img = cv2.imread(os.path.join('faces', image))
    # img = cv2.resize(img, None, fx=2, fy=2)
    height, width = img.shape[:2]
    img1 = img.copy()
    img2 = img.copy()
    img3 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # detect faces in the image
    faces1 = detector1.detect_faces(img_rgb)
    
    faces2 = detector2(gray, 2)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces3 = net.forward()
    faces4 = classifier2.detectMultiScale(img)
    
    #MTCNN
    for result in faces1:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    #DLIB    
    for result in faces2:
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        cv2.rectangle(img1, (x, y), (x1, y1), (0, 0, 255), 2)
    
    #OPENCV DNN
    for i in range(faces3.shape[2]):
        confidence = faces3[0, 0, i, 2]
        if confidence > 0.5:
            box = faces3[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
    #HAAR        
    for result in faces4:
        x, y, w, h = result
        x1, y1 = x + w, y + h
        cv2.rectangle(img3, (x, y), (x1, y1), (0, 0, 255), 2)
        
    # cv2.imwrite(os.path.join('faces', 'mtcnn', image), img)
    # cv2.imwrite(os.path.join('faces', 'dlib', image), img1)
    # cv2.imwrite(os.path.join('faces', 'dnn', image), img2)
    # cv2.imwrite(os.path.join('faces', 'haar', image), img3)
    cv2.imshow("mtcnn", img)
    cv2.imshow("dlib", img1)
    cv2.imshow("dnn", img2)
    cv2.imshow("haar", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
