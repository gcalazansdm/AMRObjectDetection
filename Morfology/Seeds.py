import cv2
import numpy as np

def watershed(image,mask):
   img_8 = np.uint8(image)
   mask_reconstructed_32 = np.int32(mask * 255)
   return cv2.watershed(img_8,mask_reconstructed_32)
