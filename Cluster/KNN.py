from sklearn.neighbors import KNeighborsClassifier
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

class KNN:
    def __init__(self,n_neighbors=5):
       self.classifier = KNeighborsClassifier(n_neighbors)

    def train(self,images,labels):
        self.classifier.fit(images, labels)

    def predict(self,X_test):
        pred_i = self.classifier.predict(X_test)
        return pred_i
    def save(self):
        with open('Resourses/KNNFile.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

def loadKNN():
    with open('Resourses/KNNFile.pkl', 'rb') as input:
        return pickle.load(input)