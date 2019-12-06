import numpy as np
import Utils

def projection(img):
    obj = img / 255.

    projH = np.sum(obj, axis=0)
    projH = Utils.normalize(projH)

    projV = np.sum(obj, axis=1)
    projV = Utils.normalize(projV)

    return  projH,projV