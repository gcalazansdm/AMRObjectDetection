import cv2
import numpy as np

def images_are_equals(imageA,imageB):
    difference = cv2.subtract(imageA,imageB,dtype = cv2.CV_32F)
    result = not np.any(difference)
    return result

def getKernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

def complement(pImage):
    ones = np.ones(pImage.shape);
    return np.subtract(ones,pImage)

def blur(img):
    dst = cv2.GaussianBlur(img,(5, 5), 0)
    return dst

def loadImg(path):
    return cv2.imread(path,cv2.CV_8SC3)

def saveImg(path,img):
    return cv2.imwrite(path,img)

def showImg(name,img):
    return cv2.imshow(name,img)

def makeMarkers(img,markers,num_seeds):
    new_images = img.copy()
    new_images[markers == -1] = [0, 0, 0]
    print(num_seeds, "sementes")

    step = 255 / num_seeds

    for i in range(0, num_seeds):
        color = round(step * i)
        new_images[markers == i] = [color * i, color * i, color * i]
    return new_images
