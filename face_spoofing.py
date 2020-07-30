import numpy as np
import cv2
from sklearn.externals import joblib
from face_detector import get_face_detector, find_faces

def calc_hist(img):
    """
    To calculate histogram of an RGB image

    Parameters
    ----------
    img : Array of uint8
        Image whose histogram is to be calculated

    Returns
    -------
    histogram : np.array
        The required histogram

    """
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

face_model = get_face_detector()
clf = joblib.load('models/face_spoofing.pkl')
cap = cv2.VideoCapture(0)

sample_number = 1
count = 0
measures = np.zeros(sample_number, dtype=np.float)

while True:
    ret, img = cap.read()
    faces = find_faces(img, face_model)

    measures[count%sample_number]=0
    height, width = img.shape[:2]
    for x, y, x1, y1 in faces:
        
        roi = img[y:y1, x:x1]
        point = (0,0)
        
        img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

        ycrcb_hist = calc_hist(img_ycrcb)
        luv_hist = calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))

        prediction = clf.predict_proba(feature_vector)
        prob = prediction[0][1]

        measures[count % sample_number] = prob

        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)

        point = (x, y-5)

        # print (measures, np.mean(measures))
        if 0 not in measures:
            text = "True"
            if np.mean(measures) >= 0.7:
                text = "False"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=img, text=text, org=point, fontFace=font, fontScale=0.9,
                            color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        
    count+=1
    cv2.imshow('img_rgb', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
