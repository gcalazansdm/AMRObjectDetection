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
    ones = np.ones(pImage.shape) * np.max(pImage)
    return np.subtract(ones,pImage)

def resize(pImage):
    height, width = pImage.shape[:2]
    ex = max( min( width, height ) / 256, 1.0 )

    if ex >= 1:
        res = cv2.resize(pImage, (round( width/ex), round(height/ex)), interpolation=cv2.INTER_CUBIC)
    else:
        res = pImage.copy()
    return res

def add(imageA,imageB):
    if(len(imageB.shape) < 3):
        imageB = cv2.cvtColor(imageB,cv2.COLOR_GRAY2RGB)
    if(len(imageA.shape) < 3):
        imageA = cv2.cvtColor(imageA,cv2.COLOR_GRAY2RGB)
    (h, w) = imageB.shape[:2]
    tempImg = Utils.normalize(imageB)
    rImg = imageA.copy()
    alpha = np.abs(1.0 - tempImg)

    print(alpha.shape)
    print(rImg.shape)
    for c in range(0, 3):
        rImg[:h, :w,  c] = imageB[:h, :w, c] + rImg[:h, :w,  c]

    return rImg

def resize_fixed_size(pImage,size):
    res = cv2.resize(pImage, size, interpolation=cv2.INTER_NEAREST)
    return res

def cleanEdges(img):
    nimg = Utils.toInt8(Utils.normalize(img))
    ret, thresh = cv2.threshold(nimg, 0, 255, cv2.THRESH_OTSU)
    no_smalls = removeSmallComponents(thresh)
    rValue = np.minimum(no_smalls,nimg)
    return rValue

def removeSmallComponents(thresh):
#    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 #   ret, thresh = cv2.threshold(grayImage,0,255,cv2.THRESH_OTSU)

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
    dst = cv2.GaussianBlur(img,(7, 7), 0)
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

def to_gray(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayImage


def img_and(img,mask):
    rImg = cv2.bitwise_and(img, mask)
    return rImg

def img_or(img,mask):
    if(len(mask.shape) < 3):
        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    print(img.shape, mask.shape)
    rImg = cv2.bitwise_or(img, mask)
    return rImg
def sub(img,mask):
    rImg = cv2.bitwise_xor(img, mask)
    rImg = cv2.bitwise_not(rImg)
    return rImg

def subtraction(img,mask):
    rImg = cv2.bitwise_and(img, cv2.bitwise_not(mask))
    return rImg