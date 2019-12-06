import numpy as np
import cv2

def mean(img,markers):
    shape = img.shape[2]
    num_seeds = np.max(markers)
    mean = np.zeros((num_seeds,shape))
    for seed in range(0,num_seeds):
        tempImg = img.copy()
        tempImg[markers != seed] = [0, 0, 0]

        for i in range(0, shape):
            mean_color = 0
            histr = cv2.calcHist([tempImg], [i], None, [256], [0, 256])
            for j in range(0,len(histr)):
                mean_color += histr[j]*j
            mean[seed,  i] += mean_color/num_seeds
    return mean