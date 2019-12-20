import Utils
import numpy as np
from Morfology.Reconstruction import reconstructionComun
import os
def reconstructionAdaptative(pImg,rage_start,max_itr=30,min_impro=1e-6):
    img = pImg
    gt=np.zeros(img.shape)
    diff=np.zeros(max_itr)
    rValue = np.zeros(max_itr)
    min_diff = np.inf
    for i in range(0,max_itr):
       recontructedImage=reconstructionComun(img,Utils.getKernel((i+rage_start-1,i+rage_start-1)))
      # Utils.saveImg(os.path.join("/home/calazans/Downloads/lol/results/53/22", "0_"+str(i)+"_mask_reconstructed.png"), recontructedImage)

       #print(np.max(recontructedImage))
       g2=np.maximum(gt,recontructedImage)
       g1=gt.copy()
       gt=g2.copy()
       diff_images = np.subtract(g2,g1)
       #print(np.max(g2))
       abs_image = np.abs(diff_images)
       diff[i]=np.max(abs_image)
       if diff[i] < min_diff:
           min_diff = diff[i]
           rValue = g2.copy()
       if diff[i] < min_impro:
           break
    return rValue