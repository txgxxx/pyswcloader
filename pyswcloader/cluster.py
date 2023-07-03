import pandas as pd
import os, sys
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

from . import distance
from . import visualization

def cluster(n_cluster=4, method='hierarchy', feature='morphology', matrix=None, data_path=None, eps=0.3, min_samples=5, **kwargs):
    if feature in ['morphology', 'precomputed']:
        info = pd.DataFrame()
        if feature=='morphology':
            matrix = distance.morphology_matrix(data_path=data_path)

        if sorted(list(matrix.index)) == sorted(list(matrix.columns)):
            vec = squareform(matrix)
            Z = linkage(vec, 'ward')
            tsne = TSNE(n_components=2, metric='precomputed', init='random', perplexity=np.min([len(matrix)-1, 30])).fit_transform(matrix)
        else:
            Z = linkage(matrix, 'ward')
            tsne = TSNE(n_components=2, perplexity=np.min([len(matrix)-1, 30])).fit_transform(matrix)

        info['file_path'] = list(matrix.index)
        info['neuron'] = [item.split('/')[-1].split('.')[0] for item in info.file_path]
        if method in ['hierarchy', 'kmeans', 'dbscan']:
            if method=='hierarchy':
                f = fcluster(Z, t=n_cluster, criterion='maxclust', **kwargs)
                plt.figure()
                dn = dendrogram(Z)
                plt.show()
                plt.close()
                info['label'] = f
            elif method=='kmeans':
                kmeans = KMeans(n_clusters=n_cluster, **kwargs)
                kmeans.fit(matrix)
                info['label'] = kmeans.labels_
            elif method=='dbscan':
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs).fit(matrix)
                info['label'] = dbscan.labels_

            sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=info.label)
        else:
            raise ValueError('method value must be hierarchy, kmeans, or dbscan.')
    else:
        raise ValueError('feature value must be either morphology or precomputed.')
    return info

def plot_cluster(info, show=True, save_path=None, **kwargs):
    for l in np.unique(info.label):
        print(l)
        file_path = list(info[info.label==l]['file_path'])
        if save_path != None:
            visualization.plot_neuron_2d(neuron_path=file_path, show=show, save_path=os.path.join(save_path, str(l)+'.png'), **kwargs)
        else:
            visualization.plot_neuron_2d(neuron_path=file_path, show=show, **kwargs)
    return True
