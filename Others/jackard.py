import os
import cv2
import numpy as np
from sklearn.metrics import jaccard_similarity_score
def loadAll(path):
    allFiles = []
    for r, d, f in os.walk(path):
        for file in f:
            allFiles.append(os.path.join(r, file))
    return allFiles

imgs = loadAll("/home/calazans/Documents/Images_pdi/imgs/gt_dir/")
dir2 = '/home/calazans/Documents/Images_pdi/imgs/Results/'
jaccard = []
for i in imgs:
    img_true = cv2.imread(i)
    img_pred = cv2.imread(os.path.join('/home/calazans/Documents/Images_pdi/imgs/Results/',os.path.basename(i)))
    img_true = np.array(img_true).ravel()
    img_pred = np.array(img_pred).ravel()
    iou = jaccard_similarity_score(img_true, img_pred)
    jaccard.append(iou)

print(np.mean(jaccard),'(',np.std(jaccard),')')
