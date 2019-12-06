import cv2
import numpy as np
import Utils
def images_are_equals(imageA,imageB):
    difference = cv2.subtract(imageA,imageB,dtype = cv2.CV_32F)
    result = not np.any(difference)
    return result

def getKernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

def complement(pImage):
    ones = np.ones(pImage.shape)
    return np.subtract(ones,pImage)

def resize(pImage):
    height, width = pImage.shape[:2]
    ex = max( min( width, height ) / 1000, 1.0 ) * 2
    print(ex)
    if ex >= 1:
        res = cv2.resize(pImage, (round( width/ex), round(height/ex)), interpolation=cv2.INTER_CUBIC)
    else:
        res = pImage.copy()
    return res

def resize_fixed_size(pImage,size):
    res = cv2.resize(pImage, (size, size), interpolation=cv2.INTER_CUBIC)
    return res

def removeSmallComponents(img):
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayImage,0,255,cv2.THRESH_OTSU)

    connectivity = 8

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity,cv2.CV_32S)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 150

    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

def getConnectedObjects(img):
    connectivity = 8

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(Utils.toInt8(img), connectivity,cv2.CV_32S)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 150
    imgs = []

    for i in range(0, nb_components):
        img2 = np.zeros((output.shape))
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
            imgs.append(img2)
    return imgs

def blur(img):
    dst = cv2.GaussianBlur(img,(5, 5), 0)
    return dst

def loadImg(path):
    return cv2.imread(path)

def saveImg(path,img):
    return cv2.imwrite(path,img)

def showImg(name,img):
    return cv2.imshow(name,img)

def makeMarkers(img,markers,num_seeds):
    new_images = img.copy()
    new_images[markers == -1] = [0, 0, 0]

    step = 255 / num_seeds

    for i in range(0, num_seeds):
        color = round(step * i)
        new_images[markers == i] = [color * i, color * i, color * i]
    return new_images

def binarize(image):
    _,rValue = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    return rValue

def binarize_otsu(image):
    _,rValue = cv2.threshold(image, 10, 255, cv2.THRESH_OTSU)
    return rValue

def mkborder(img):
    borderType = cv2.BORDER_CONSTANT
    shape = img.shape
    top = int(0.05 * shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * shape[1])  # shape[1] = cols
    right = left

    dst = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, [0,0,0])
    return dst , (top,bottom,left,right)

def crop(img,pos):
    x,y = img.shape[:2]
    crop_img = img[pos[0]:x-pos[1],pos[2]:y-pos[2]]
    return crop_img

def fill(img):
    # Copy the thresholded image.
    rValue = img.copy()

    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(rValue, mask, (0, 0), (255,255,255))

    return rValue

def img_and(img,mask):
    print(img.shape)
    print(type(img))
    print(mask.shape)
    print(type(mask))
    rImg = cv2.bitwise_and(img, mask)
    return rImg
def sub(img,mask):
    rImg = cv2.bitwise_xor(img, mask)
    rImg = cv2.bitwise_not(rImg)
    return rImg