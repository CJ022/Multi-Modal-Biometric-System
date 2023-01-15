import numpy as np
import cv2
import os
from datetime import datetime

import faceRecognition as fr
print (fr)

test_img=cv2.imread(r'C:\Users\246687\Desktop\BE Final Year Project\Face-Recognition_lpbh\face_recognition\Test images\Test_image3.jpg')      #Give path to the image which you want to test


faces_detected,gray_img=fr.faceDetection(test_img)
print("face Detected: ",faces_detected)


face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#use of r- normal string as a path is converted
face_recognizer.read(r"C:\Users\246687\Desktop\BE Final Year Project\Face-Recognition_lpbh\trainingData.yml")  #Give path of where trainingData.yml is saved

name={0:"Atharva"}            #  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1. 


def markAttendance(name):
    with open('attend1.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # if name not in nameList:  
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S:%D')
            f.writelines(f'\n{name},{dtString}')

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print ("Confidence :",confidence)
    print("label :",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence<60):
        fr.put_text(test_img,"unknown",x,y);
        break;
    else:
        markAttendance(predicted_name);
        fr.put_text(test_img,predicted_name,x,y);

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
