#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from util.camera import camera
import python.darknet as dn
from python.darknet import detect


# In[ ]:





# In[2]:


weights = 'yolov4.weights'
netcfg  = 'cfg/yolov4.cfg'
data = 'cfg/coco.data'


# In[ ]:


net  = dn.load_net(netcfg.encode('utf-8'), weights.encode('utf-8'), 0)
meta = dn.load_meta(data.encode('utf-8'))


# In[ ]:

cam = 0
# cam = 'rtsp://admin:qwer1234@192.168.88.249:554/Streaming/channels/1'
# cap = camera(cam)
# print(f"Camera is alive?: {cap.p.is_alive()}")
cap = cv2.VideoCapture(cam)


# In[7]:


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# In[8]:


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],            detection[2][1],            detection[2][2],            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


# In[ ]:

n = 0 
while True:
    try:
        # frame = cap.get_frame()
        _,frame = cap.read()
        # W,H = frame.shape[:2]
        # print(W,H)
        detected = detect(net,meta,frame,thresh=0.5)
        frame = cvDrawBoxes(detected, frame)
        cv2.waitKey(1)
        cv2.imshow('frame',frame)
    except Exception as e:
        print(e)


# In[ ]:




