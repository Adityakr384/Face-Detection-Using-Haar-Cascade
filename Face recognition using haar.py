import cv2
import os
import numpy as np


def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    face_haar_cascade=cv2.CascadeClassifier('c:/Users/Aditya/.spyder-py3/haarcascade_frontalface_default.xml')#Load haar classifier
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles

    return faces,gray_img

#This module takes images stored in disk,and performs face recognition
test_img=cv2.imread('C:/Users/Aditya/.spyder-py3/testing/27/SarE_02317_f_22_i_nf_nc_hp_2016_1_e0_nl_o.jpg')#test_img path
face,gray_img=faceDetection(test_img)
print("faces_detected:",face)


if len(face)==1:
    (x,y,w,h)=face[0]
    roi_gray=gray_img[y:y+h,x:x+h]
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,255,0),thickness=5)
    resized_img=cv2.resize(test_img,(200,200))
    cv2.imshow("face detection",resized_img)
    cv2.waitKey(0)#Waits until a key is pressed
cv2.destroyAllWindows()
