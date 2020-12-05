# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:01:06 2020

@author: XZ-WAVE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                                'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                                'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
                                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def detect_obj(img_path,model='fast_rcnn',threshold=0.9,
               rect_th=4,text_size=0.85,text_th=2,plot_obj_conf=False,decimals=3,
               box_color='blue',text_color='green',custom_box_color=None,custom_text_color=None):
    '''
    function that performs object detection on an image using a pre-trained Fast-RCNN or Mask-RCNN model from Pytorch
    img_path: string - path to img that you want to perform detection on 
    model: string - either fast_rcnn or mask_rcnn only for now
    threshold: float - a detection confidence threshold. Default 0.9 (Bounding Boxes appear only when model is 90% confident in the object or higher)
    rect_th: int/float - rectangle thickness in pixels (see opencv library's cv2.rectangle method for more details)
    text_size: int/float - rectangle thickness in pixels (see opencv library's cv2.putText method for more details)
    text_th: int/float - rectangle thickness in pixels (see opencv library's cv2.putText method for more details)
    plot_obj_conf: bool - plot the confidence next to the class above the bounding box during a detection 
    decimals: int - only relevant when plot_obj_conf is set to True - confidence is rounded to the decimal argument provided by user
    box_color: string - red,green,blue,custom, - if custom, user provides a 3-tuple of the form (R,G,B) for specific color
    text_color: string - red,green,blue,custom - if custom, user provides a 3-tuple of the form (R,G,B) for specific color
    '''
    if box_color == 'red':
        color_box = (255,0,0)
    if box_color == 'green':
        color_box = (0,255,0)
    if box_color == 'blue':
        color_box = (0,0,225)
    if box_color == 'custom':
        color_box = custom_box_color
        
    if text_color == 'red':
        color_text = (255,0,0)
    if text_color == 'green':
        color_text = (0,255,0)
    if text_color == 'blue':
        color_text = (0,0,225)
    if text_color == 'custom':
        color_text = custom_text_color
    if model == 'fast_rcnn':
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    if model == 'mask_rcnn':
        model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    img = Image.open(img_path) # Load the image
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    pred = model([img]) #need to get softmax outputs 
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Scores
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    print('Model successfully loaded! Detecting objects...')
    for i in range(len(pred_boxes)):
        print('{}/{} objects detected'.format(i+1,len(pred_boxes)))
        cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=color_box, thickness=rect_th) # Draw Rectangle with the coordinates
        if plot_obj_conf == True:
           confidence = np.format_float_positional(pred_score[i],precision=3)
           cv2.putText(img, "{} conf: {}".format(pred_class[i],confidence), pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, color=color_text,thickness=text_th) # Write the predictionclass
        if plot_obj_conf == False:
           cv2.putText(img, pred_class[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, color=color_text,thickness=text_th)
    plt.figure(figsize=(30,30)) # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()
        
    return pred_boxes, pred_class


    