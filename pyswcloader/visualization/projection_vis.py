import os
import distinctipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import colormaps
from itertools import groupby
import pandas as pd
from scipy.stats import linregress
import seaborn as sns
from pyswcloader.reader import brain


def plot_topographic_projection(data, threshold=0.05, save=False, save_path=os.getcwd()):
    terminal_groups = data.groupby('soma_region')
    corr = pd.DataFrame(index=data.soma_region.unique(), columns=['r_value', 'p_value', 'soma_region'])
    for region, group in terminal_groups:
        corr.loc[region, 'r_value'] = linregress(group.soma_pca, group.term_pca).rvalue
        corr.loc[region, 'p_value'] = linregress(group.soma_pca, group.term_pca).pvalue
    showdata = corr.dropna()
    showdata.r_value = abs(corr.r_value)
    showdata = showdata[showdata.p_value < threshold]
    cmap = colormaps.get_cmap('coolwarm_r')
    norm = colors.Normalize(vmin=min(showdata.p_value), vmax=max(showdata.p_value))
    bar = cm.ScalarMappable(norm, cmap)
    norm_values = (showdata.p_value - min(showdata.p_value)) / (max(showdata.p_value) - min(showdata.p_value))
    showdata.color = [colormaps.get_cmap('coolwarm_r')(value) for value in norm_values]
    plt.figure(figsize=(25, 5))
    plt.bar(x=showdata.index.astype(str), height=showdata.r_value, color=showdata.color, width=0.5)
    plt.colorbar(bar)
    plt.ylim((0, 1))
    if save:
        plt.savefig(os.path.join(save_path, 'topographic_projection.png'))
    plt.close()
    return


def plot_correlation(data, region, save=False, save_path=os.getcwd()):
    region_info = data[data.soma_region == region]
    x = region_info.soma_pca
    y = region_info.term_pca
    regression_result = linregress(x, y)
    sns.scatterplot(x, y, s=10)
    plt.plot([min(x), max(x)],
             [min(x) * regression_result.slope + regression_result.intercept,
              max(x) * regression_result.slope + regression_result.intercept],
             c='black',
             linewidth=1,
             label="$R$=%.3f" % regression_result.rvalue + ", $P$=%.3e" % regression_result.pvalue
             )
    plt.legend(loc='upper right', prop={'size': 10})
    plt.title(region)
    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('')
    plt.yticks([])
    if save:
        plt.savefig(os.path.join(save_path, 'topographic_correlation.png'))
    plt.close()
    return


def plot_allen_template_clustermap(axon_length, cluster_results, with_dendrogram=False, linkage=None, save=False, save_path=os.getcwd()):
    projection = axon_length.loc[cluster_results.neuron]
    projection_t = projection.loc[:, projection.any()].T
    region_info = pd.read_csv(brain.FILTER_REGION_PATH)
    region_info = region_info.loc[region_info.mark == 'bottom']
    labels = list(cluster_results.label.unique())
    label_colors = distinctipy.get_colors(len(labels), pastel_factor=0.65)
    label_colors = dict(zip(map(int, labels), label_colors))
    col_colors = projection.labels.map(label_colors)
    projection_t = pd.concat([projection_t, region_info], axis=1, join='inner')
    data_agg = projection_t[:-3].groupby('parent').agg('sum')
    data_agg = np.log(data_agg)
    data_agg = data_agg.replace(-np.inf, 0)
    data_agg.index.name = ''
    distribution = np.sum(data_agg, axis=1)
    distribution = distribution.sort_values(ascending=False)
    top_region = distribution.shape[0] if distribution[0] < 30 else 31
    region_dict_family = region_info[['parent', 'family']].drop_duplicates()
    region_dict_family = region_dict_family.set_index('parent', drop=False)
    region_dict_family = region_dict_family.loc[distribution.index[:top_region]]
    data_family = pd.concat([data_agg, region_dict_family], axis=1, join='inner').sort_values(by='family')
    data_family.index.name = ''
    family_colors = sns.hls_palette(len(set(data_family.family)), s=0.45)
    row_family = dict(zip(map(str, list(set(data_family.family))), family_colors))
    row_colors = data_family.family.map(row_family)
    row_colors.rename('region', inplace=True)
    row_colors.index.name = ''
    width = max(25, int(len(region_dict_family) * 0.6))
    length = max(width, int(len(cluster_results) * 0.01))
    if with_dendrogram:
        g = sns.clustermap(data_agg.loc[region_dict_family.index.tolist()], cmap="coolwarm",
                           row_colors=row_colors, col_colors=col_colors,
                           row_cluster=False, col_cluster=with_dendrogram,
                           col_linkage=linkage,
                           dendrogram_ratio=(.1, .2),
                           # cbar_pos=(.02, .6, .02, .2),
                           linewidths=.0001,
                           figsize=(length, width),
                           colors_ratio=0.02,
                           xticklabels=False,
                           cbar_kws={"pad": 0.001, "use_gridspec": False})
    else:
        g = sns.clustermap(data_agg.loc[region_dict_family.index.tolist()], cmap="coolwarm",
                           row_colors=row_colors, col_colors=col_colors,
                           row_cluster=False, col_cluster=with_dendrogram,
                           dendrogram_ratio=(0.1, 0.05),
                           # cbar_pos=(.02, .6, .02, .2),
                           linewidths=.0001,
                           figsize=(length, width),
                           colors_ratio=0.02,
                           xticklabels=False,
                           cbar_kws={"pad": 0.001, "use_gridspec": False})
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
    borders_col = np.cumsum([0] + [sum(1 for i in g) for k, g in groupby(col_colors.values)])
    col_labels = projection.loc[neu_names].labels.unique()
    for start, end, l in zip(borders_col[:-1], borders_col[1:], col_labels):
        g.ax_col_colors.text((start + end) / 2, -0.01, l, color='black', ha='center', va='bottom',
                             transform=g.ax_col_colors.get_xaxis_transform(), size=21)
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, 'projection_pattern.png'))
    return


def plot_customized_template_clustermap(axon_length,
                                        cluster_results,
                                        with_dendrogram=False,
                                        linkage=None,
                                        save=False,
                                        save_path=os.getcwd()):
    projection = axon_length.loc[cluster_results.neuron]
    projection_t = projection.loc[:, projection.any()].T
    projection_t = np.log(projection_t)
    projection_t = projection_t.replace(-np.inf, 0)
    labels = list(cluster_results.label.unique())
    label_colors = distinctipy.get_colors(len(labels), pastel_factor=0.65)
    label_colors = dict(zip(map(int, labels), label_colors))
    col_colors = projection.labels.map(label_colors)
    distribution = np.sum(projection_t, axis=1)
    distribution = distribution.sort_values(ascending=False)
    region_num = distribution.shape[0] if distribution[0] < 30 else 30
    data_t = projection_t.loc[distribution.index[:region_num]]
    width = max(25, int(region_num * 0.6))
    length = max(width, int(len(cluster_results) * 0.01))
    if with_dendrogram:
        g = sns.clustermap(data_t, cmap="coolwarm",
                           col_colors=col_colors,
                           row_cluster=False, col_cluster=with_dendrogram,
                           col_linkage=linkage,
                           dendrogram_ratio=(.1, .2),
                           cbar_pos=(.01, .6, .02, .2),
                           linewidths=.0001,
                           figsize=(length, width),
                           colors_ratio=0.015,
                           xticklabels=False,
                           cbar_kws={"pad": 0.001}
                           )
    else:
        g = sns.clustermap(data_t, cmap="coolwarm",
                           col_colors=col_colors,
                           row_cluster=False,
                           col_cluster=with_dendrogram,
                           cbar_pos=(.01, .6, .02, .2),
                           linewidths=.0001,
                           figsize=(length, width),
                           dendrogram_ratio=(0.1, 0.05),
                           colors_ratio=0.02,
                           xticklabels=False,
                           cbar_kws={"pad": 0.001}
                           )
    g.ax_heatmap.yaxis.tick_left()
    g.ax_heatmap.tick_params(left=False)

    neu_names = data_t.columns.values
    if with_dendrogram:
        leaves = g.dendrogram_col.reordered_ind
        neu_names = neu_names[leaves]
        col_colors = col_colors.loc[neu_names]
    borders_col = np.cumsum([0] + [sum(1 for i in g) for k, g in groupby(col_colors.values)])
    col_labels = projection.loc[neu_names].labels.unique()
    for start, end, l in zip(borders_col[:-1], borders_col[1:], col_labels):
        g.ax_col_colors.text((start + end) / 2, -0.01, l, color='black', ha='center', va='bottom',
                             transform=g.ax_col_colors.get_xaxis_transform(), size=21)
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, 'projection_pattern.png'))
    return











