from sklearn.cluster import SpectralClustering
import Utils

def cluster(img,grups):
    normalizedimg = Utils.normalize(img)
    spectral = SpectralClustering(n_clusters=grups,eigen_solver='amg')

    grp = spectral.fit_predict(normalizedimg)
    return grp