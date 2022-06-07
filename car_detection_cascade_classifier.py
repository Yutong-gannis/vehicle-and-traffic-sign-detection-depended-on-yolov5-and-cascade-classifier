# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:09:05 2022

@author: lenovo
"""

import numpy as np
import cv2

def getROI(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    triangle = np.array([[(int(width*0), int(height*0.8)), (width, int(height*0.8)), (int(width*0.7), int(height*0.42)), (int(width*0.3), int(height*0.42))]])
    black_image = np.zeros_like(img)
    mask = cv2.fillPoly(black_image, triangle, 255)
    masked_image = cv2.bitwise_and(img, mask)
    # cv2.imshow('roi', masked_image)
    return masked_image

def findcars(img, gray, car_cascade):
    car = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)  # 参数：1、灰度图片， 2、缩放比例， 3、阈值
    for (x, y, w, h) in car:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 1)
    cv2.imshow('Video Cam',img)

def findlight(img, gray, trafficlight_cascade):
    trafficlight = trafficlight_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)  # 参数：1、灰度图片， 2、缩放比例， 3、阈值
    for (x, y, w, h) in trafficlight:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)
    cv2.imshow('Video Cam',img)
    
def parameters():
    testvideo_path = 'test_video/9.mp4'
    car_cascade_path = 'cascade classifier/cars.xml'
    trafficlight_cascade = 'cascade classifier/TrafficLight_HAAR_16Stages.xml'
    return testvideo_path, car_cascade_path, trafficlight_cascade
    
def run():
    testvideo_path, car_cascade_path, trafficlight_cascade = parameters()
    camera = cv2.VideoCapture(testvideo_path)
    cv2.namedWindow('Video Cam', cv2.WINDOW_NORMAL)
    car_cascade = cv2.CascadeClassifier(car_cascade_path)
    trafficlight_cascade = cv2.CascadeClassifier(trafficlight_cascade)

    while cv2.waitKey(1) != 27:
        success, img = camera.read()
        img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_AREA)
        gray = getROI(img)
        findcars(img, gray, car_cascade)
        findlight(img, gray, trafficlight_cascade)
        
    camera.release()
    cv2.destroyAllWindows()
    
run()