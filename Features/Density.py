import numpy as np
import Utils
import cv2
def calculateDensity(image):
    temp_image = Utils.normalize(image)
    x,y = temp_image.shape[:2]
    sum = np.sum(temp_image,axis=1)
    sum = np.sum(sum,axis=0)
    sum = sum /(x*y)
    return sum
