import cv2
def humoments(im):
    # Calculate Moments
    moments = cv2.moments(im)

    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    return huMoments