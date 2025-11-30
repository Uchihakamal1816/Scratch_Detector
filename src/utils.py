import cv2
import numpy as np

def mask_to_bboxes(mask):
    mask = mask.astype("uint8")
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        boxes.append([x,y,x+w,y+h])
    return boxes
