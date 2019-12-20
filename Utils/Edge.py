import cv2
import numpy as np
from Utils.GeneralUtils import normalize
import Morfology
import scipy.ndimage.filters
from Utils.OpenCVUtils import getKernel
def Test(img):
    return cv2.Canny(img,100,200)
def sobel(img,propH=.01,propW=.01):
    dilated_img = Morfology.dilatate(img, getKernel((3,3)))
    imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(imgLab)
    b_lines = np.power(sobel_inner(b),2)
    g_lines = np.power(sobel_inner(g),2)
    r_lines = np.power(sobel_inner(r),2)
    bg = np.sum([b_lines,g_lines],axis=0)
    bgr_lines = np.sum([bg,r_lines],axis=0)
    o4 = np.sqrt(bgr_lines)

    return normalize(o4)*255
def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0],
                     [1, 1, 1],
                     [0, 0, 0]])
    winSE = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    winS = np.array([[0, 1, 0],
                     [0, 1, 0],
                     [0, 1, 0]])
    winSW = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag
def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

def sobel_inner(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = blur(img)
    #gray = blur(gray)
    o1 = cv2.Sobel(img, cv2.CV_64F, 1, 0,ksize=3)
    o2 = cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=3)
    mag = np.hypot(o1, o2)
    ang = np.arctan2(o2, o1)
    # threshold
    fudgefactor = 0.5
    threshold = 4 * fudgefactor * np.mean(mag)
    mag[mag < threshold] = 0
    # non-maximal suppression
    mag = orientated_non_max_suppression(mag, ang)
    #mag[mag > 0] *= 255
    return normalize(mag)

#TEste Internet
'''
import cv2
import numpy as np
import scipy.ndimage.filters

def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0],
                     [1, 1, 1],
                     [0, 0, 0]])
    winSE = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    winS = np.array([[0, 1, 0],
                     [0, 1, 0],
                     [0, 1, 0]])
    winSW = np.array([[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

# compute sobel response
sobelx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)'''
'''mag = np.hypot(sobelx, sobely)
ang = np.arctan2(sobely, sobelx)
# threshold
fudgefactor = 0.5
threshold = 4 * fudgefactor * np.mean(mag)
mag[mag < threshold] = 0
# non-maximal suppression
mag = orientated_non_max_suppression(mag, ang)
# alternative but doesn't consider gradient direction
# mag = skimage.morphology.thin(mag.astype(np.bool)).astype(np.float32)

# create mask
mag[mag > 0] = 255
mag = mag.astype(np.uint8)
'''