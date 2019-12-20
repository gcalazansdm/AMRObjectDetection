import cv2
import numpy as np
import Utils

def calculateBorders(img):
    img_8 = Utils.toInt8(img)
    contours, hierarchy = cv2.findContours(img_8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    hull = cv2.convexHull(cnt)

    epsilon = 0.1 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    return approx

def findContours(img):
    img_8 = Utils.toInt8(img)
    contours, hierarchy = cv2.findContours(img_8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    canvas = np.zeros(img.shape, np.uint8)
    cnt = contours[0]
    hull = cv2.convexHull(cnt)

    epsilon = 0.1 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    cv2.drawContours(canvas, [approx], -1, (255, 0, 0), 3)
    return canvas