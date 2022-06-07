# vehicle-and-traffic-sign-detection-depended-on-yolov5-and-cascade-classifier
I use both yolov5 and cascade classifier to detect vehicle and traffic sign based on bdd100k dataset. The reference time of yolov5 is about 17ms much better than cascade classifier., which is enough for real-time detection.

### cascade classifier
Cascade classifier file to detect cars, pedestrian, stoplights, two wheelers.

### data 
.yaml file for datasets path

### datasets
The bdd100k datain yolo's format

### test_video
Some videos for testing

### weights
yolov5n.engine for inferencing on tensorrt
yolov5n_4000_640.pt is trained from 4000 pictures of 640x640 pixels
yolov5n_6000_640.pt is trained from 6000 pictures of 640x640 pixels
yolov5n_8000_416.pt is trained from 8000 pictures of 416x416 pixels
yolov5s_3000_640.pt is trained from 3000 pictures of 640x640 pixels
yolov5s_5000_640.pt is trained from 5000 pictures of 640x640 pixels

### convert2yolo.py
convert bdd100k labels to yolo

### car_detection_yolov5.py
The main program to detect vehicles by yolov5. Change your own parameters in parameters().

car_detection_cascade_classifier.py
The main program to detect vehicles by cascade classifier. Change your own parameters in parameters().