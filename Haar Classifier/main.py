import cv2
import numpy as np
from keras.models import load_model

model = load_model(r"mask_detector.model")
results = {1:'without mask', 0:'mask'}
GR_dict = {1:(0,0,255), 0:(0,255,0)}
rect_size = 4
cap = cv2.VideoCapture(0) 
haarcascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cap.read()
    if(rval == False):
        break

    im=cv2.flip(im, 1) 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rerect_size = cv2.resize(gray, (gray.shape[1] // rect_size, gray.shape[0] // rect_size))
    faces = haarcascade.detectMultiScale(rerect_size)
    # print(len(faces))
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
        face_img = im[y:y+h, x:x+w]
        try:
            cv2.imshow("Face", face_img)
        except:
            pass
        rerect_sized = cv2.resize(face_img,(224,224))
        normalized = rerect_sized/255.0
        reshaped = np.reshape(normalized, (1,224,224,3))
        reshaped = np.vstack([reshaped])

        result = model.predict(reshaped)
        # print(result)
        label = np.argmax(result, axis=1)[0]
      
        cv2.rectangle(im, (x,y), (x+w,y+h), GR_dict[label], 2)
        cv2.rectangle(im, (x,y-40), (x+w,y), GR_dict[label], -1)
        cv2.putText(im, results[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(im, "No. of faces detected: {0}".format(len(faces)), (0, im.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('LIVE', im)
    key = cv2.waitKey(10)
    if key == 113: 
        break
cap.release()
cv2.destroyAllWindows()