import numpy as np
import os
import cv2
from skimage import color

def toInt8(img):
    info = normalize(img)
    data = 255 * info  # Now scale by 255
    img2 = data.astype(np.uint8)
    return img2
def normalize(array):
    min_val = np.min(array)
    temp_val = array - min_val
    max_val = np.max(temp_val)
    if(max_val != 0):
        temp_val = temp_val / max_val
    return temp_val
def getDir(maindir,newdir):
    dir = os.path.join(maindir,newdir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
def print_(img):
    print(np.max(img),np.min(img))

def loadAll(path):
    allFiles = []
    for r, d, f in os.walk(path):
        for file in f:
            allFiles.append(os.path.join(r, file))
    return allFiles
def bgrToLab(img):
    print(img.shape)
    m_rgb = img[..., ::-1]
    lab = color.rgb2lab(m_rgb)
    return lab