import Utils
import numpy as np
import cv2
from Morfology.GeodesicMethods import geoDilatate
from Morfology.GeodesicMethods import geoErode
def reconstructionComun(pImg,kernel):
    img = (pImg)

    #primeira parte com dilatate
    img=Utils.complement(img)
    eroded_image=cv2.erode(img,kernel)
    image_reconstructed=morfologicalreconstruction(eroded_image,img,kernel)
    image_reconstructed_complement=Utils.complement(image_reconstructed)

    #parte com erode
    eroded_image=cv2.erode(image_reconstructed_complement,kernel)
    image_reconstructed=morfologicalreconstruction(eroded_image,image_reconstructed_complement,kernel)#,mode='erode')
    image_reconstructed_complement=Utils.complement(image_reconstructed)
    image_reconstructed_complement=Utils.complement(image_reconstructed_complement)

    return image_reconstructed_complement


def morfologicalreconstruction(mask,pimage,kernel,mode='dilate'):
    image = pimage
    rValue = image
    actual_mask = mask
    prevImage = np.zeros(image.shape)
    while(not Utils.images_are_equals(rValue,prevImage)):
        prevImage = rValue
        if(mode=='dilate'):
            actual_mask = geoDilatate(actual_mask,image,kernel)
            rValue = np.minimum(actual_mask,image)
        else:
            actual_mask = geoErode(actual_mask,image,kernel)
            rValue = np.maximum(actual_mask,image)
    return rValue