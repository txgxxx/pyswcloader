import argparse
import codecs
from itertools import groupby
from multiprocessing import Pool
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.pyplot import gcf
from seaborn import clustermap, hls_palette
import distinctipy
import pandas as pd
import os
import numpy as np
import json
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from pyswcloader.reader import brain, swc, io


# def read_allen_region_info():
#     region_info = pd.read_csv(os.path.join('pyswcloader/database', 'region_dict.csv'), index_col=0)
#     # region_info = region_info.loc[(region_info.mark == 'bottom') & (region_info.family != 'root')]
#     return region_info

def get_soma_info(data_path, region_info):
    regions = []
    opts = []
    for p in data_path:
        soma_info = swc.swc_preprocess(p).loc[1]
    #     opts.append(([soma_info.x, soma_info.y, soma_info.z], io.ALLEN_ANNOTATION, 10))
    # pool = Pool()
    # results = pool.map(brain.find_region, opts)
        region_index = brain.find_region([soma_info.x, soma_info.y, soma_info.z], io.ALLEN_ANNOTATION, 10)
        soma_region = io.STL_ACRO_DICT[region_index] if region_index in io.STL_ACRO_DICT else 'None'
        soma_region = region_info.loc[soma_region, 'parent'] if soma_region != 'None' else 'None'
        regions.append(soma_region)
    return regions


def read_pfc_info(

            projection: pd.DataFrame = None,
            save: bool = False,
            save_path: str = os.getcwd(),
            data_path: str = None,
            json_path:str = None,
            **kwargs):
    info = pd.DataFrame()
    with codecs.open(json_path, 'r', 'utf-8-sig') as f:
        data = json.load(f)

        info['neuron'] = [values['file'].replace('_', '-').split('.')[0] for key, values in data['neuron_data'].items()]
        info['file_path'] = [os.path.join(data_path, item + '.swc') for item in info['neuron']]
        info.set_index('neuron', inplace=True)


        info['label'] = [int(values['class']) for key, values in data['neuron_data'].items()]
        region_info = pd.read_csv(os.path.join('pyswcloader/database', 'region_dict.csv'), index_col=0)
        info['soma_info'] = get_soma_info(info['file_path'], region_info)
        fig = plot_allen_template_clustermap(projection, info,
                                            save=save,
                                            save_path=save_path)

        if save:
            info.to_csv(os.path.join(save_path, 'cluster_results.csv'))
        return info




def plot_allen_template_clustermap(axon_length,

                                   cluster_results,
                                   with_dendrogram=False,
                                   linkage=None,
                                   save=False,
                                   save_path=os.getcwd()):

    if not with_dendrogram:
        cluster_results = cluster_results.sort_values(by='label')

    projection = axon_length.loc[cluster_results.index]
    projection_t = projection.loc[:, projection.any()].T
    region_info = brain.read_allen_region_info()


    soma_unique = list(cluster_results.soma_info.unique())
    label_colors = distinctipy.get_colors(len(soma_unique), pastel_factor=0.65)
    label_colors = dict(zip(soma_unique, label_colors))
    col_colors = pd.DataFrame(index=cluster_results.index)
    col_colors['soma_region'] = cluster_results.soma_info.map(label_colors)

    labels = list(cluster_results.label.unique())
    label_colors1 = distinctipy.get_colors(len(labels), pastel_factor=0.65)
    label_colors1 = dict(zip(map(int, labels), label_colors1))
    col_colors['label'] = cluster_results.label.map(label_colors1)




    projection_t = pd.concat([projection_t, region_info], axis=1, join='inner')
    data_agg = projection_t[:-3].groupby('parent').agg('sum')
    data_agg.index.name = ''
    distribution = np.mean(data_agg, axis=1)
    distribution = distribution.sort_values(ascending=False)

    const_top_region = 30
    top_region = min(distribution.shape[0], const_top_region) if distribution[
                                                                     0] < const_top_region else const_top_region
    region_dict_family = region_info[['parent', 'family']].drop_duplicates()
    region_dict_family = region_dict_family.set_index('parent', drop=False)
    region_dict_family = region_dict_family.loc[distribution.index[:top_region]]
    region_dict_family = region_dict_family.sort_values(by='family')

    data_family = pd.concat([data_agg, region_dict_family], axis=1, join='inner').sort_values(by='family')
    data_family.index.name = ''
    family_colors = hls_palette(len(set(data_family.family)), s=0.45)
    row_family = dict(zip(map(str, list(set(data_family.family))), family_colors))

    row_colors = data_family.family.map(row_family)
    row_colors.rename('region', inplace=True)
    row_colors.index.name = ''

    width = max(25, int(len(region_dict_family) * 0.6))
    length = max(30, int(len(cluster_results) * 0.01))

    data_log = np.log(data_agg)
    data_log = data_log.replace(-np.inf, 0)

    fig = plt.figure()
    common_params = {
        'cmap': "coolwarm",
        'col_colors': col_colors,
        'row_colors': row_colors,
        'row_cluster': False,
        'col_cluster': with_dendrogram,
        'linewidths': 0.0001,
        'figsize': (length, width),
        'xticklabels': False,
        'colors_ratio': 0.02,
        'cbar_kws': {"pad": 0.001, "use_gridspec": False}
    }

    if with_dendrogram:
        dendrogram_params = {
            'col_linkage': linkage,
            'dendrogram_ratio': (0.1, 0.2),
        }
    else:
        dendrogram_params = {
            'dendrogram_ratio': (0.1, 0.05),

        }
    g = clustermap(data_log.loc[region_dict_family.index.tolist()], **common_params, **dendrogram_params)

    g.ax_heatmap.yaxis.tick_left()
    g.ax_heatmap.tick_params(left=False)
    g.fig.subplots_adjust(left=0.)
    g.ax_cbar.set_position((.01, .6, .02, .2))
    row_colors = row_colors.loc[region_dict_family.index.tolist()]
    borders = np.cumsum([0] + [sum(1 for i in g) for k, g in groupby(row_colors.values)])
    family_region = region_dict_family.family.unique()
    for start, end, l in zip(borders[:-1], borders[1:], family_region):
        g.ax_row_colors.text(-0.06, (start + end) / 2, l, color=row_family[l], ha='right', va='center',
                             transform=g.ax_row_colors.get_yaxis_transform(), size=21)
    neu_names = data_agg.columns.values
    if with_dendrogram:
        leaves = g.dendrogram_col.reordered_ind
        neu_names = neu_names[leaves]
        col_colors = col_colors.loc[neu_names]
        cluster_results = cluster_results.loc[neu_names]
    handles = [Patch(facecolor=label_colors[soma_info], edgecolor='black', label=soma_info) for soma_info in soma_unique]
    plt.legend(handles, soma_unique, title='soma region', bbox_to_anchor=(1.01, 0.8), bbox_transform=plt.gcf().transFigure)
        # for neu in neu_names:
        #     g.ax_col_dendrogram.bar(0, 0, color=col_colors[neu], label=col_colors, linewidth=0);
        # l1 = g.ax_col_dendrogram.legend(title='Network', loc="center", ncol=5, bbox_to_anchor=(0.35, 0.89),
        #                                 bbox_transform=gcf().transFigure)

        # g.ax_col_dendrogram.legend(title='soma info', loc='right', ncol=2)
    # borders_col = np.cumsum([0] + [sum(1 for i in g) for k, g in groupby(col_colors.values)])
    # col_labels = cluster_results.label.unique()
    # for start, end, l in zip(borders_col[:-1], borders_col[1:], col_labels):
    #     g.ax_col_colors.text((start + end) / 2, -0.01, l, color='black', ha='center', va='bottom',
    #                          transform=g.ax_col_colors.get_xaxis_transform(), size=21)
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, 'projection_pattern.png'), bbox_inches='tight')
        data_agg.loc[region_dict_family.index.tolist()].to_csv(os.path.join(save_path, 'projection_pattern.csv'))

    return fig


def cluster(n_cluster: int = 64,
            metric: str = 'scores',
            matrix: pd.DataFrame = None,
            projection: pd.DataFrame = None,
            save: bool = False,
            save_path: str = os.getcwd(),
            data_path: str = None,
            **kwargs) -> pd.DataFrame:

    info = pd.DataFrame()

    if sorted(list(matrix.index)) == sorted(list(matrix.columns)):
        if metric != 'nblast':
            vec = squareform(np.nan_to_num(matrix.values))
        else:
            vec = matrix
        Z = linkage(vec, 'ward')
        # tsne = TSNE(n_components=2, metric='precomputed', init='random',
        #             perplexity=np.min([len(matrix) - 1, 30])).fit_transform(matrix)
    else:
        Z = linkage(matrix, 'ward')
        # tsne = TSNE(n_components=2, perplexity=np.min([len(matrix) - 1, 30])).fit_transform(matrix)


    info['neuron'] = [item.split('/')[-1].split('.')[0] for item in matrix.index]
    info['file_path'] = [os.path.join(data_path, item + '.swc') for item in info['neuron']]
    info.set_index('neuron', inplace=True)

    f = fcluster(Z, t=n_cluster, criterion='maxclust', **kwargs)
    info['label'] = f
    region_info = pd.read_csv(os.path.join('pyswcloader/database', 'region_dict.csv'), index_col=0)
    info['soma_info'] = get_soma_info(info['file_path'], region_info)


    fig = plot_allen_template_clustermap(projection, info,
                                         with_dendrogram=True,
                                         linkage=Z,
                                         save=save,
                                         save_path=save_path)

    if save:
        info.to_csv(os.path.join(save_path, 'cluster_results.csv'))
    return info


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="SingleNeuronAnalyseSummary args description")
    parse.add_argument('--data_path', '-d', type=str, default='', help='swc data path')
    parse.add_argument('--axon_length', '-a', type=str, default='', help='swc data path')
    parse.add_argument('--score_path', '-s', type=str, default='', help='swc data path')
    parse.add_argument('--metric', '-m', type=str, default='', help="results save path")
    parse.add_argument('--save_path', '-p', type=str, default='', help="results save path")
    parse.add_argument('--nclusters', '-n', type=int, default='', help="results save path")
    args = parse.parse_args()
    projection = pd.read_csv(args.axon_length, index_col=0)
    # if 'nblast' in args.score_path:
    #     score = 1-pd.read_csv(args.score_path, index_col=0)
    # else:
    score = pd.read_csv(args.score_path, index_col=0)
    cluster(n_cluster=args.nclusters,
        matrix=score,
            data_path=args.data_path,
            metric=args.metric,
           projection=projection,
           save=True,
           save_path=args.save_path)
    # read_pfc_info(projection=projection,
    #               save_path=args.save_path,
    #               save=True,
    #               data_path=args.data_path,
    #               json_path='/home/dongzhou/data/mouse/scores/PFC_cluster_result/pfc_neuron.info')
