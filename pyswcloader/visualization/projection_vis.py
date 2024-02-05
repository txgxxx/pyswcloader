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
from seaborn import scatterplot, hls_palette, clustermap
from sklearn.decomposition import PCA

from projection import projection_batch, projection_neuron
from reader import brain, io


def plot_topographic_projection(data, template=brain.Template.allen, threshold=10, p_threshold=0.05, save=False, save_path=os.getcwd()):
    data['soma_pca'] = PCA(n_components=1).fit_transform(data[['soma_x', 'soma_y', 'soma_z']])
    data['term_pca'] = PCA(n_components=1).fit_transform(data[['x', 'y', 'z']])
    terminal_groups = data.groupby('region')
    corr = pd.DataFrame(index=data.index.unique(), columns=['r_value', 'p_value'])
    for region, group in terminal_groups:
        if len(group) > threshold:
            corr.loc[region, 'r_value'] = linregress(group.soma_pca, group.term_pca).rvalue
            corr.loc[region, 'p_value'] = linregress(group.soma_pca, group.term_pca).pvalue
    showdata = corr.dropna()
    showdata.r_value = abs(corr.r_value)
    showdata = showdata[showdata.p_value < p_threshold]
    cmap = colormaps.get_cmap('coolwarm_r')
    norm = colors.Normalize(vmin=min(showdata.p_value), vmax=max(showdata.p_value))
    bar = cm.ScalarMappable(norm, cmap)
    norm_values = (showdata.p_value - min(showdata.p_value)) / (max(showdata.p_value) - min(showdata.p_value))
    showdata['color'] = [colormaps.get_cmap('coolwarm_r')(value) for value in norm_values]
    plt.figure(figsize=(25, 5))
    if template == brain.Template.allen:
        region_info = brain.read_allen_region_info()
        showdata['family'] = [region_info.loc[item, 'family'] for item in showdata.index]
        showdata = showdata.sort_values(by='family')
        vlinelist = showdata.family
        start = -0.5
        mark = vlinelist[0]
        for item in vlinelist[1:]:
            start += 1
            if item != mark:
                plt.vlines(x=start, ymin=0, ymax=1, color='gray', linestyles='dashed')
            mark = item
    plt.bar(x=showdata.index.astype(str), height=showdata.r_value, color=showdata.color, width=0.5)
    plt.colorbar(bar)
    plt.ylim((0, 1))
    if save:
        plt.savefig(os.path.join(save_path, 'topographic_projection.png'))
    plt.close()
    return showdata


def plot_correlation(data, region, save=False, save_path=os.getcwd()):
    region_data = data[data.region == region]
    region_data['soma_pca'] = PCA(n_components=1).fit_transform(region_data[['soma_x', 'soma_y', 'soma_z']])
    region_data['term_pca'] = PCA(n_components=1).fit_transform(region_data[['x', 'y', 'z']])
    x = region_data.soma_pca
    y = region_data.term_pca
    regression_result = linregress(x, y)
    scatterplot(x, y, s=10)
    plt.plot([min(x), max(x)],
             [min(x) * regression_result.slope + regression_result.intercept,
              max(x) * regression_result.slope + regression_result.intercept],
             c='black',
             linewidth=1,
             label="$R$=%.3f" % regression_result.rvalue + ", $P$=%.3e" % regression_result.pvalue
             )
    plt.legend(loc='upper right', prop={'size': 10})
    plt.title(region)
    plt.xlabel('soma')
    plt.xticks([])
    plt.ylabel('term')
    plt.yticks([])
    if save:
        plt.savefig(os.path.join(save_path, region + '_topographic_correlation.png'))
    plt.close()
    return


def plot_allen_template_clustermap(axon_length, cluster_results, with_dendrogram=False, linkage=None, save=False, save_path=os.getcwd()):
    if not with_dendrogram:
        cluster_results = cluster_results.sort_values(by='label')
    projection = axon_length.loc[cluster_results.index]
    projection_t = projection.loc[:, projection.any()].T
    region_info = brain.read_allen_region_info()
    labels = list(cluster_results.label.unique())
    label_colors = distinctipy.get_colors(len(labels), pastel_factor=0.65)
    label_colors = dict(zip(map(int, labels), label_colors))
    col_colors = cluster_results.label.map(label_colors)
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
    region_dict_family = region_dict_family.sort_values(by='family')
    data_family = pd.concat([data_agg, region_dict_family], axis=1, join='inner').sort_values(by='family')
    data_family.index.name = ''
    family_colors = hls_palette(len(set(data_family.family)), s=0.45)
    row_family = dict(zip(map(str, list(set(data_family.family))), family_colors))
    row_colors = data_family.family.map(row_family)
    row_colors.rename('region', inplace=True)
    row_colors.index.name = ''
    width = max(25, int(len(region_dict_family) * 0.6))
    length = max(width, int(len(cluster_results) * 0.01))
    if with_dendrogram:
        g = clustermap(data_agg.loc[region_dict_family.index.tolist()], cmap="coolwarm",
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
        g = clustermap(data_agg.loc[region_dict_family.index.tolist()], cmap="coolwarm",
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
        cluster_results = cluster_results.loc[neu_names]
    borders_col = np.cumsum([0] + [sum(1 for i in g) for k, g in groupby(col_colors.values)])
    col_labels = cluster_results.label.unique()
    for start, end, l in zip(borders_col[:-1], borders_col[1:], col_labels):
        g.ax_col_colors.text((start + end) / 2, -0.01, l, color='black', ha='center', va='bottom',
                             transform=g.ax_col_colors.get_xaxis_transform(), size=21)
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, 'projection_pattern.png'))
        data_agg.loc[region_dict_family.index.tolist()].to_csv(os.path.join(save_path, 'projection_pattern.csv'))
    return


def plot_customized_template_clustermap(axon_length,
                                        cluster_results,
                                        with_dendrogram=False,
                                        linkage=None,
                                        save=False,
                                        save_path=os.getcwd()):
    if not with_dendrogram:
        cluster_results = cluster_results.sort_values(by='label')
    projection = axon_length.loc[cluster_results.index]
    projection_t = projection.loc[:, projection.any()].T
    projection_t = np.log(projection_t)
    projection_t = projection_t.replace(-np.inf, 0)
    labels = list(cluster_results.label.unique())
    label_colors = distinctipy.get_colors(len(labels), pastel_factor=0.65)
    label_colors = dict(zip(map(int, labels), label_colors))
    col_colors = cluster_results.label.map(label_colors)
    distribution = np.sum(projection_t, axis=1)
    distribution = distribution.sort_values(ascending=False)
    region_num = distribution.shape[0] if distribution[0] < 30 else 30
    data_t = projection_t.loc[distribution.index[:region_num]]
    width = max(25, int(region_num * 0.6))
    length = max(width, int(len(cluster_results) * 0.01))
    if with_dendrogram:
        g = clustermap(data_t, cmap="coolwarm",
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
        g = clustermap(data_t, cmap="coolwarm",
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
    col_labels = cluster_results.label.unique()
    for start, end, l in zip(borders_col[:-1], borders_col[1:], col_labels):
        g.ax_col_colors.text((start + end) / 2, -0.01, l, color='black', ha='center', va='bottom',
                             transform=g.ax_col_colors.get_xaxis_transform(), size=21)
    if save:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path, 'projection_pattern.png'))
        data_t.to_csv(os.path.join(save_path, 'projection_pattern.csv'))
    return











