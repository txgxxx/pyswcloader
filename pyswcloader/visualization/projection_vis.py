import os
import distinctipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from itertools import groupby
import pandas as pd
from scipy.stats import linregress
from seaborn import scatterplot, hls_palette, clustermap
from sklearn.decomposition import PCA

from pyswcloader.reader import brain


# from ..reader import brain


def plot_topographic_projection(data, template=brain.Template.allen, threshold=10, p_threshold=0.05, save=False, show=False, save_path=os.getcwd()):
    terminal_groups = data.groupby('region')
    corr = pd.DataFrame(index=data.index.unique(), columns=['r_value', 'p_value'])
    for region, group in terminal_groups:
        topo_info = group[['x', 'y', 'z', 'neuron', 'soma_x', 'soma_y', 'soma_z']].groupby('neuron').agg('mean')
        topo_info['soma_pca'] = PCA(n_components=1).fit_transform(topo_info[['soma_x', 'soma_y', 'soma_z']])
        topo_info['term_pca'] = PCA(n_components=1).fit_transform(topo_info[['x', 'y', 'z']])
        if len(topo_info) > threshold:
            corr.loc[region, 'r_value'] = linregress(topo_info.soma_pca, topo_info.term_pca).rvalue
            corr.loc[region, 'p_value'] = linregress(topo_info.soma_pca, topo_info.term_pca).pvalue
    showdata = corr.dropna()
    showdata.r_value = abs(corr.r_value)
    showdata = showdata[showdata.p_value < p_threshold]
    cmap = cm.get_cmap('coolwarm_r')
    norm = colors.Normalize(vmin=min(showdata.p_value), vmax=max(showdata.p_value))
    bar = cm.ScalarMappable(norm, cmap)
    norm_values = (showdata.p_value - min(showdata.p_value)) / (max(showdata.p_value) - min(showdata.p_value))
    showdata['color'] = [cm.get_cmap('coolwarm_r')(value) for value in norm_values]
    fig = plt.figure(figsize=(25, 5))
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
    if show:
        plt.show()
    return showdata, fig


def plot_correlation(data, region, save=False, show=False, save_path=os.getcwd()):
    region_data = data[data['region'] == region]
    region_data = region_data[['x', 'y', 'z', 'neuron', 'soma_x', 'soma_y', 'soma_z']].groupby('neuron').agg('mean')
    region_data['soma_pca'] = PCA(n_components=1).fit_transform(region_data[['soma_x', 'soma_y', 'soma_z']])
    region_data['term_pca'] = PCA(n_components=1).fit_transform(region_data[['x', 'y', 'z']])
    x = region_data.soma_pca
    y = region_data.term_pca
    regression_result = linregress(x, y)
    fig = plt.figure()
    scatterplot(x=x, y=y, s=10)
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
    if show:
        plt.show()
    return fig


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
    length = max(width, int(len(cluster_results) * 0.01))

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

    return fig


def plot_customized_template_clustermap(axon_length,
                                        cluster_results,
                                        with_dendrogram=False,
                                        linkage=None,
                                        save=False,
                                        save_path=os.getcwd()):
    # 数据验证


    cluster_results = cluster_results.sort_values(by='label')
    projection = axon_length.loc[cluster_results.index]

    projection_t = projection.loc[:, projection.any()].T
    labels = list(cluster_results.label.unique())


    label_colors = distinctipy.get_colors(len(labels), pastel_factor=0.65)
    label_colors = dict(zip(map(int, labels), label_colors))
    col_colors = cluster_results.label.map(label_colors)

    distribution = np.mean(projection_t, axis=1)
    distribution = distribution.sort_values(ascending=False)
    region_num = min(distribution.shape[0], 30)

    data_t = projection_t.loc[distribution.index[:region_num]]
    width = max(25, int(region_num * 0.6))
    length = max(width, int(len(cluster_results) * 0.01))

    data_log = np.log(data_t)
    data_log = data_log.replace(-np.inf, 0)

    fig = plt.figure()
    # 绘图参数
    common_params = {
        'cmap': "coolwarm",
        'col_colors': col_colors,
        'row_cluster': False,
        'cbar_pos': (0.01, 0.6, 0.02, 0.2),
        'linewidths': 0.0001,
        'figsize': (length, width),
        'xticklabels': False,
        'cbar_kws': {"pad": 0.001}
    }

    if with_dendrogram:
        dendrogram_params = {
            'col_cluster': with_dendrogram,
            'col_linkage': linkage,
            'dendrogram_ratio': (0.1, 0.2),
            'colors_ratio': 0.015
        }
    else:
        dendrogram_params = {
            'col_cluster': False,
            'dendrogram_ratio': (0.1, 0.05),
            'colors_ratio': 0.02
        }
    g = clustermap(data_log, **common_params, **dendrogram_params)

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
    return fig











