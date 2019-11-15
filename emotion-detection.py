import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

#model can be loaded
new_model=load_model('../model.h5')

face_cascade = cv2.CascadeClassifier('C:\\Users\\hp\\Desktop\\open cv\\DATA\\haarcascades\\haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_detected= face_cascade.detectMultiScale(gray_img,scaleFactor=1.32, minNeighbors=5)
    
    for (x,y,w,h) in face_detected:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),thickness=4)
        roi_gray=gray_img[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels=image.img_to_array(roi_gray)
        img_pixels=np.expand_dims(img_pixels,axis=0)
        img_pixels=img_pixels/255
        
        predictions=new_model.predict(img_pixels)
        
        max_index=np.argmax(predictions[0])
        
        emotions= ('Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise')
        predicted_emotion=emotions[max_index]
        
        cv2.putText(frame,predicted_emotion,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
    
    resized_img=cv2.resize(frame,(1000,700))
    cv2.imshow('Facial Emotion Analysis',resized_img)
    
    if cv2.waitKey(10) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()