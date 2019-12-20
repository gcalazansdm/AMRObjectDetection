import numpy as np
import cv2
import Utils
import os
def crop(img,pos):
    x,y = img.shape[:2]
    crop_img = img[pos[0]:x-pos[1],pos[2]:y-pos[2]]
    return crop_img

def mkborder(img):
    borderType = cv2.BORDER_CONSTANT
    shape = img.shape
    top = int(0.05 * shape[0])  # shape[0] = rows
    bottom = top
    left = int(0.05 * shape[1])  # shape[1] = cols
    right = left

    dst = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, [0,0,0])
    return dst , (top,bottom,left,right)

def multiFill(img):
    top = img.copy()
    x,y = img.shape[:2]
    bottom = img.copy()
    left = img.copy()
    right = img.copy()

    top[0, :] = 255
    bottom[x-1, :] = 255
    left[:, 0] = 255
    right[:, y-1] = 255
    top = fillandcrop(top)
    bottom = fillandcrop(bottom)
    left = fillandcrop(left)
    right = fillandcrop(right)
    o1 = cv2.bitwise_and(top,bottom)
    o2 = cv2.bitwise_and(left,right)
    o3 = cv2.bitwise_and(fillandcrop(o1),fillandcrop(o2))
    rValue = cv2.erode(o3,Utils.getKernel((3,3)))
    rValue = cv2.dilate(rValue,Utils.getKernel((3,3)))

    return rValue

def fillandcrop(img):
    rImg, pos = mkborder(img)
    rImg = fill(rImg)
    rImg = crop(rImg,pos)
    return rImg

def fill(img):
    # Copy the thresholded image.
    rValue = img.copy()

    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(rValue, mask, (0, 0), (255,255,255))

    return rValue