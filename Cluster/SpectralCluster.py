from sklearn.cluster import SpectralClustering
import numpy as np
import Utils

def cluster(img,grups):
    normalizedimg = Utils.normalize(img)
  #  threeDImg = np.
    # calculate the affinity / similarity matrix
    # compute the degree matrix
    # compute the normalized laplacian / affinity matrix (method 1)

    # perform the eigen value decomposition
    # select k largest eigen vectors
    # perform kmeans clustering on the matrix U
    spectral = SpectralClustering(n_clusters=grups,eigen_solver='lobpcg')

    grp = spectral.fit_predict(normalizedimg)
    return grp