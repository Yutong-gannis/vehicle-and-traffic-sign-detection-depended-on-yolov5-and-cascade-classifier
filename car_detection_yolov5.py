# -*- coding: utf-8 -*-
"""
Created on Wed May  4 01:11:27 2022

@author: lenovo
"""

import numpy as np
import cv2
import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, time_sync
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

def getROI(img):
    '''
    mask data if you only need to detect some area of the image
    Parameters
    ----------
    img : the original image

    Returns
    -------
    masked_image : the image have been masked
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    triangle = np.array([[(int(width*0), int(height*0.8)), (width, int(height*0.8)), (int(width*1), int(height*0.3)), (int(width*0), int(height*0.3))]])
    black_image = np.zeros_like(img)
    mask = cv2.fillPoly(black_image, triangle, 255)
    masked_image = cv2.bitwise_and(img, mask)
    # cv2.imshow('roi', masked_image)
    return masked_image

def findcars(img, img_size, model, classes, device):
    imgsz=(img_size, img_size)
    imgsz = check_img_size(imgsz, s=model.stride)
    
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    model.warmup(imgsz=(1 if model.pt else bs, 3, *imgsz))
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    t1 = time_sync()
    # gray = getROI(img)
    gray = img
    gray = torch.from_numpy(gray).to(device)
    gray = gray.half() if model.fp16 else gray.float()  # uint8 to fp16/32
    gray /= 255
    with torch.no_grad():
        if len(gray.shape) == 3:
            gray = gray[None]
        gray = gray.transpose(3,2).transpose(2,1)
        t2 = time_sync()
        dt[0] += t2 - t1
        
        results = model(gray, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2
        
        results = non_max_suppression(results, model.conf, model.iou)
        dt[2] += time_sync() - t3
    
    t4 = time_sync()
    font=cv2.FONT_HERSHEY_SIMPLEX
    for n, det in enumerate(results):
        seen += 1
        det = det.cpu().numpy()
        for i in range(len(det)):
            if int(det[i][5]) == 2: # car
                m = (det[i][2]+det[i][0])/2
                if det[i][3]-det[i][1] >= 120 and m>280 and m<360:
                    cv2.rectangle(img, (int(det[i][0]), int(det[i][1])), (int(det[i][2]), int(det[i][3])), (0, 0, 255), 1)
                    cv2.putText(img,  '{} {:.3f}'.format(classes[int(det[i][5])],det[i][4]),(int(det[i][0]),int(det[i][1])-3),font,0.3, (0, 0, 255), 1)
                else:
                    cv2.rectangle(img, (int(det[i][0]), int(det[i][1])), (int(det[i][2]), int(det[i][3])), (0, 255, 0), 1)
                    cv2.putText(img,  '{} {:.3f}'.format(classes[int(det[i][5])],det[i][4]),(int(det[i][0]),int(det[i][1])-3),font,0.3, (0, 255, 0), 1)
            
            elif int(det[i][5]) == 0: # person
                if det[i][3]-det[i][1] >= 120:
                    cv2.rectangle(img, (int(det[i][0]), int(det[i][1])), (int(det[i][2]), int(det[i][3])), (0, 0, 255), 1)
                    cv2.putText(img,  '{} {:.3f}'.format(classes[int(det[i][5])],det[i][4]),(int(det[i][0]),int(det[i][1])-3),font,0.3, (0, 0, 255), 1)
                else:
                    cv2.rectangle(img, (int(det[i][0]), int(det[i][1])), (int(det[i][2]), int(det[i][3])), (0, 191, 255), 1)
                    cv2.putText(img,  '{} {:.3f}'.format(classes[int(det[i][5])],det[i][4]),(int(det[i][0]),int(det[i][1])-3),font,0.3, (0, 191, 255), 1)
            
            else:
                cv2.rectangle(img, (int(det[i][0]), int(det[i][1])), (int(det[i][2]), int(det[i][3])), (0, 245, 255), 1)
                cv2.putText(img,  '{} {:.3f}'.format(classes[int(det[i][5])],det[i][4]),(int(det[i][0]),int(det[i][1])-3),font,0.3, (0, 245, 255), 1)
    cv2.imshow('Video Cam',img)
    
    dt[3] += time_sync() - t4
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms show, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    return results

def parameters():
    '''
    setect parameters

    Returns
    -------
    testvideo_path : .mp4
        path of test videos
    weight_path : .pt
        path of weight
    data_path : .taml
        path of data
    confidence : float
        confidence to detection objects
    iou : float
        iou to confirm the same object
    classes : list
        classes
    img_size : int
        image size

    '''
    testvideo_path = '9.mp4'
    weight_path = 'weights\yolov5n_6000.pt'
    data_path = r"data\cocobdd.yaml"
    confidence = 0.3
    iou = 0.5
    classes = ["person", "rider", "car", "bus", "truck", "bike", "motor", "tl_green", "tl_red", "tl_yellow", "tl_none", "traffic sign", "train"]
    img_size = 640
    return testvideo_path, weight_path, data_path, confidence, iou, classes, img_size

def run():
    testvideo_path, weight_path, data_path, confidence, iou, classes, img_size = parameters()
    camera = cv2.VideoCapture(testvideo_path)
    cv2.namedWindow('Video Cam', cv2.WINDOW_NORMAL)
    device = select_device('0')
    model = DetectMultiBackend(weight_path, device=device, dnn=False, data=data_path, fp16=False)
    model.warmup()
    model.conf = confidence
    model.iou = iou

    while cv2.waitKey(1) != 27:
        success, img = camera.read()
        if success:
            img = cv2.resize(img, (img_size,img_size), interpolation=cv2.INTER_AREA)
            results = findcars(img, img_size, model, classes, device)
    
    camera.release()
    cv2.destroyAllWindows()
    
run()