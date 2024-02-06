import os
from enum import Enum
import pandas as pd
import numpy as np

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from seaborn import scatterplot

import distance
import reader.io
from reader import brain
from visualization import projection_vis, neuron_vis


class Method(Enum):
    hierarchy = 1
    kmeans = 2
    dbscan = 3


class Feature(Enum):
    morphology = 1
    precomputed = 2


def cluster(n_cluster: int = 4,
            template: brain.Template = brain.Template.allen,
            method: Method = Method.hierarchy,
            feature: Feature = Feature.morphology,
            matrix: pd.DataFrame = None,
            projection: pd.DataFrame = None,
            data_path: str = None,
            eps: float = 0.3,
            min_samples: int = 5,
            save: bool = False,
            save_path: str = os.getcwd(),
            **kwargs) -> pd.DataFrame:
    """
    :param n_cluster:
    :param template:
    :param method:
    :param feature:
    :param matrix:
    :param projection: axon_length info
    :param data_path:
    :param eps:
    :param min_samples:
    :param save:
    :param save_path:
    :param kwargs:
    :return:
    """
    info = pd.DataFrame()
    if feature == Feature.morphology:
        matrix = distance.morphology_matrix(data_path=data_path, save_path=os.path.join(save_path, 'scores_record.txt'))
    if sorted(list(matrix.index)) == sorted(list(matrix.columns)):
        vec = squareform(matrix)
        Z = linkage(vec, 'ward')
        tsne = TSNE(n_components=2, metric='precomputed', init='random',
                    perplexity=np.min([len(matrix) - 1, 30])).fit_transform(matrix)
    else:
        Z = linkage(matrix, 'ward')
        tsne = TSNE(n_components=2, perplexity=np.min([len(matrix) - 1, 30])).fit_transform(matrix)

    info['file_path'] = list(matrix.index)
    info['neuron'] = [item.split('/')[-1].split('.')[0] for item in info.file_path]
    info.set_index('neuron', inplace=True)
    if method == Method.hierarchy:
        f = fcluster(Z, t=n_cluster, criterion='maxclust', **kwargs)
        info['label'] = f
    elif method == Method.kmeans:
        kmeans = KMeans(n_clusters=n_cluster, **kwargs)
        kmeans.fit(matrix)
        info['label'] = kmeans.labels_
    elif method == Method.dbscan:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, **kwargs).fit(matrix)
        info['label'] = dbscan.labels_
    fig = plt.figure()
    plot = scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=info.label)
    if projection is not None:
        if template == brain.Template.allen:
            projection_vis.plot_allen_template_clustermap(projection, info,
                                                         with_dendrogram=(method == Method.hierarchy),
                                                         linkage=Z,
                                                         save=save,
                                                         save_path=save_path)
        else:
            projection_vis.plot_customized_template_clustermap(projection, info,
                                                              with_dendrogram=(method == Method.hierarchy),
                                                              linkage=Z,
                                                              save=save,
                                                              save_path=save_path)
    if save:
        info.to_csv(os.path.join(save_path, 'cluster_results.csv'))
        fig.savefig(os.path.join(save_path, 'tsne.png'))
    return info


def plot_cluster(info, show=True, save_path=None, **kwargs):
    for l in np.unique(info.label):
        file_path = list(info[info.label == l]['file_path'])
        if save_path is not None:
            neuron_vis.plot_neuron_2d(neuron_path=file_path, show=show,
                                         save_path=os.path.join(save_path, 'cluster-' + str(l) + '.png'), **kwargs)
        else:
            neuron_vis.plot_neuron_2d(neuron_path=file_path, show=show, **kwargs)
    return True

if __name__ == '__main__':
    projection = pd.read_csv('/home/cdc/data/mouse_data/temp/axon_length.csv', index_col=0)
    cluster(
        n_cluster=5,
        projection=projection,
        save=True,
        data_path='/home/cdc/data/mouse_data/test',
        save_path='/home/cdc/data/mouse_data/temp/'
    )
