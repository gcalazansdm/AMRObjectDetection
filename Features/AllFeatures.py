from Features.Projection import projection
from Features.Density import calculateDensity as density
from Features.Contours import calculateBorders as cBorder
import numpy as np
import Utils

def all_features_batch(images):
    features = []
    for image in images:
        af = all_features(image)
        features.append(af)
    return np.array(features)

def all_features(image):
    pImage = Utils.resize_fixed_size(image,256)
    x,y = projection(pImage)
    borders = cBorder(image)
    array = np.concatenate([x, y], axis=None)
    array = np.concatenate([array,density(image)], axis=None)
    array = np.concatenate([array,borders.shape], axis=None)
    return array
