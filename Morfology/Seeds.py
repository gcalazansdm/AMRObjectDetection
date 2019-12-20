import cv2
import numpy as np
from Morfology.watershed.Watershed import Watershed
def watershed(image,mask):
  # img_8 = np.uint8(image)
  # mask_reconstructed_32 = np.int32(mask * 255)
   w = Watershed()
   return w.apply(mask)