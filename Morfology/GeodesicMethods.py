import cv2
import  numpy as np

def dilatate(mask,kernel):
    return cv2.dilate(mask,kernel)

def geoDilatate(mask,image,kernel):
    actual_mask = cv2.dilate(mask,kernel)
    return np.minimum(actual_mask,image)

def geoErode(mask,image,kernel):
    actual_mask = cv2.erode(mask,kernel)
    return np.maximum(actual_mask,image)