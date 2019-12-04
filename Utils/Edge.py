import cv2
import numpy as np
from Utils.GeneralUtils import normalize
from Utils.OpenCVUtils import blur

def sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = blur(gray)
    gray = blur(gray)
    o1 = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    o2 = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    o1_2= np.power(o1,2)
    o2_2= np.power(o2,2)
    o3 = np.sum([o1_2,o2_2],axis=0)
    o4 = np.sqrt(o3)
    return normalize(o4)