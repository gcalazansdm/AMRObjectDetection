import numpy as np
import os
import Utils
import Features
import cv2
from Cluster.KNN import KNN

def calculate(path):
    imgs_path = Utils.loadAll(path)
    imgs = []
    for i in range(0,len(imgs_path)):
        img = Utils.loadImg(imgs_path[i])
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    sum = Features.all_features_batch(imgs)
    return sum

valuesDocs = calculate('/home/calazans/Documents/lista 08/temp/Documentos/')
labelsDocs = np.ones(valuesDocs.shape[0])

valuesNotDocs = calculate('/home/calazans/Documents/lista 08/temp/Bin/')
labelsNotDocs = np.zeros(valuesNotDocs.shape[0])

array = np.concatenate([valuesNotDocs[0:120], valuesDocs[0:120]], axis=0)
labels = np.concatenate([labelsDocs[0:120], labelsNotDocs[0:120]], axis=0)
tarray = np.concatenate([valuesNotDocs[120:], valuesDocs[120:]], axis=0)
tlabels = np.concatenate([labelsDocs[120:], labelsNotDocs[120:]], axis=0)
error = []
classifier = KNN(3)
classifier.train(array, labels)
classifier.save()
for i in range(1, 40):
    classifier = KNN(i)
    classifier.train(array,labels)
    predicted = classifier.predict(tarray)
    error.append(np.mean(predicted != tlabels))
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()