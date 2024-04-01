import glob
import os
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly as py
import plotly.graph_objects as go
import plotly.figure_factory as ff
import distinctipy
from scipy.spatial.distance import squareform

from plotly.subplots import make_subplots

from pyswcloader.reader.brain import *


# from ..reader.brain import Template, read_allen_region_info


class _Dendrogram(object):
    """Refer to FigureFactory.create_dendrogram() for docstring."""

    def __init__(
        self,
        X,
        orientation="bottom",
        labels=None,
        color='black',
        width=np.inf,
        height=np.inf,
        xaxis="xaxis",
        yaxis="yaxis",
        hovertext=None,
    ):
        self.orientation = orientation
        self.labels = labels
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.data = []
        self.leaves = []
        self.color =color
        self.sign = {self.xaxis: 1, self.yaxis: 1}
        self.layout = {self.xaxis: {}, self.yaxis: {}}

        if self.orientation in ["left", "bottom"]:
            self.sign[self.xaxis] = 1
        else:
            self.sign[self.xaxis] = -1

        if self.orientation in ["right", "bottom"]:
            self.sign[self.yaxis] = 1
        else:
            self.sign[self.yaxis] = -1



        (dd_traces, xvals, yvals, ordered_labels, leaves) = self.get_dendrogram_traces(
            X,  hovertext)

        self.labels = ordered_labels
        self.leaves = leaves
        yvals_flat = yvals.flatten()
        xvals_flat = xvals.flatten()

        self.zero_vals = []

        for i in range(len(yvals_flat)):
            if yvals_flat[i] == 0.0 and xvals_flat[i] not in self.zero_vals:
                self.zero_vals.append(xvals_flat[i])

        if len(self.zero_vals) > len(yvals) + 1:
            # If the length of zero_vals is larger than the length of yvals,
            # it means that there are wrong vals because of the identicial samples.
            # Three and more identicial samples will make the yvals of spliting
            # center into 0 and it will accidentally take it as leaves.
            l_border = int(min(self.zero_vals))
            r_border = int(max(self.zero_vals))
            correct_leaves_pos = range(
                l_border, r_border + 1, int((r_border - l_border) / len(yvals))
            )
            # Regenerating the leaves pos from the self.zero_vals with equally intervals.
            self.zero_vals = [v for v in correct_leaves_pos]

        self.zero_vals.sort()
        self.layout = self.set_figure_layout(width, height)
        self.data = dd_traces

    def set_axis_layout(self, axis_key):
        """
        Sets and returns default axis object for dendrogram figure.

        :param (str) axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.
        :rtype (dict): An axis_key dictionary with set parameters.

        """
        axis_defaults = {
            "type": "linear",
            "ticks": "outside",
            "mirror": "allticks",
            "rangemode": "tozero",
            "showticklabels": False,
            "zeroline": False,
            "showgrid": False,
            "showline": True,
        }

        if len(self.labels) != 0:
            axis_key_labels = self.xaxis
            if self.orientation in ["left", "right"]:
                axis_key_labels = self.yaxis
            if axis_key_labels not in self.layout:
                self.layout[axis_key_labels] = {}
            self.layout[axis_key_labels]["tickvals"] = [
                zv * self.sign[axis_key] for zv in self.zero_vals
            ]
            self.layout[axis_key_labels]["ticktext"] = self.labels
            self.layout[axis_key_labels]["tickmode"] = "array"

        self.layout[axis_key].update(axis_defaults)

        return self.layout[axis_key]

    def set_figure_layout(self, width, height):
        """
        Sets and returns default layout object for dendrogram figure.

        """
        self.layout.update(
            {
                "showlegend": False,
                "autosize": False,
                "hovermode": "closest",
                "width": width,
                "height": height,
            }
        )

        self.set_axis_layout(self.xaxis)
        self.set_axis_layout(self.yaxis)

        return self.layout

    def get_dendrogram_traces(
        self, Z, hovertext
    ):
        """
        Calculates all the elements needed for plotting a dendrogram.

        :param (ndarray) X: Matrix of observations as array of arrays
        :param (list) colorscale: Color scale for dendrogram tree clusters
        :param (function) distfun: Function to compute the pairwise distance
                                   from the observations
        :param (function) linkagefun: Function to compute the linkage matrix
                                      from the pairwise distances
        :param (list) hovertext: List of hovertext for constituent traces of dendrogram
        :rtype (tuple): Contains all the traces in the following order:
            (a) trace_list: List of Plotly trace objects for dendrogram tree
            (b) icoord: All X points of the dendrogram tree as array of arrays
                with length 4
            (c) dcoord: All Y points of the dendrogram tree as array of arrays
                with length 4
            (d) ordered_labels: leaf labels in the order they are going to
                appear on the plot
            (e) P['leaves']: left-to-right traversal of the leaves

        """
        P = dendrogram(
            Z,
            orientation=self.orientation,
            labels=self.labels,
            no_plot=True,
        )

        icoord = np.array(P["icoord"])
        dcoord = np.array(P["dcoord"])
        ordered_labels = np.array(P["ivl"])


        trace_list = []

        for i in range(len(icoord)):
            # xs and ys are arrays of 4 points that make up the '∩' shapes
            # of the dendrogram tree
            if self.orientation in ["top", "bottom"]:
                xs = icoord[i]
            else:
                xs = dcoord[i]

            if self.orientation in ["top", "bottom"]:
                ys = dcoord[i]
            else:
                ys = icoord[i]
            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]
            trace = dict(
                type="scatter",
                x=np.multiply(self.sign[self.xaxis], xs),
                y=np.multiply(self.sign[self.yaxis], ys),
                mode="lines",
                marker=dict(color=self.color),
                text=hovertext_label,
                hoverinfo="text",
            )

            try:
                x_index = int(self.xaxis[-1])
            except ValueError:
                x_index = ""

            try:
                y_index = int(self.yaxis[-1])
            except ValueError:
                y_index = ""

            trace["xaxis"] = "x" + x_index
            trace["yaxis"] = "y" + y_index

            trace_list.append(trace)

        return trace_list, icoord, dcoord, ordered_labels, P["leaves"]


def get_web_cluster_info(cluster_info):
    cluster_group = cluster_info.groupby(['label', 'neuron', 'file_path']).count()
    classes = 'class="table table-border"'
    cluster_html = cluster_group.to_html(header=False)
    cluster_html = cluster_html.replace('class="dataframe"', classes)
    # bootstrap_link = '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">'
    # cluster_html = bootstrap_link + '\n' + cluster_html
    return cluster_html

def get_web_neuron_plot(neuron_path):
    path_list = glob.glob(os.path.join(os.path.abspath(neuron_path), 'cluster-*.png'), recursive=True)
    # img_html = '<div><div class="listyle">\n'
    # for path in path_list:
    #     img_html += '<li><img src="file://%s" alt="%s" /></li>\n'%(path, path.split('/')[-1].split('.')[0])
    # img_html += '</div></div>'
    img_html = '<div class="container-fluid"><div class="row justify-content-start">\n'

    for idx, path in enumerate(path_list):
        img_html += '<div class="col-xs-6 col-md-3 col-lg-2 "> <div class="thumbnail">' \
                    '<img class="img-responsive" src="%s" alt="%s" /><p class="text-center">%s</p></div></div>\n'%(path.split('/')[-1], path.split('/')[-1].split('.')[0], path.split('/')[-1].split('\\')[-1].split('.')[0])
    img_html += '</div></div>'
    return img_html


def get_web_projection_info(cluster_info, scores, axon_length, template):
    vec = squareform(scores)
    Z = linkage(vec, 'ward')
    neus = [f.split('/')[-1].split('.')[0] for f in scores.index]
    den = _Dendrogram(
        Z,
        orientation='bottom',
        labels=neus)
    den_fig = go.Figure(data=den.data, layout=den.layout)
    axon_length = np.log(axon_length)
    axon_length = axon_length.replace(-np.inf, 0)
    neu_names = axon_length.columns.values
    leaves = den.leaves
    neu_names = neu_names[leaves]
    axon_length = axon_length[neu_names]
    labels = cluster_info.loc[neu_names].label.values
    label_colors = distinctipy.get_colors(len(np.unique(labels)), pastel_factor=0.7)
    label_colors = ['rgb(' + str(int(c[0] * 255)) + ',' + str(int(c[1] * 255)) + ',' + str(int(c[2] * 255)) + ')' for c
                    in label_colors]
    labels_range = np.linspace(0, 1, len(label_colors))
    label_colors = list(map(list, zip(labels_range, label_colors)))
    labels_map = ff.create_annotated_heatmap(z=[cluster_info.loc[neu_names].T.loc['label']],
                                             x=den_fig['layout']['xaxis']['tickvals'],
                                             customdata=[neu_names],
                                             hovertemplate=
                                             '<b>neuron</b>: %{customdata}' +
                                             '<br><b>label</b>: %{z}<extra></extra>',
                                             colorscale=label_colors,
                                             annotation_text=None)
    labels_fig = go.Figure(labels_map)
    labels_fig.update_layout(xaxis_showticklabels=False, yaxis_showticklabels=False)

    legend_anno = None
    legend_fig = None
    if template == Template.allen:
        region_info = read_allen_region_info()
        row_family = region_info.loc[axon_length.index]
        family_colors = distinctipy.get_colors(len(set(row_family.family)), pastel_factor=0.7)
        family_colors = ['rgb(' + str(int(c[0] * 255)) + ',' + str(int(c[1] * 255)) + ',' + str(int(c[2] * 255)) + ')'
                         for c in family_colors]
        family_range = np.linspace(0, 1, len(set(row_family.family)))
        family_colors = list(map(list, zip(family_range, family_colors)))
        family_index = dict(zip(list(set(row_family.family)), list(range(len(set(row_family.family))))))
        family_region = [region_info.loc[reg, 'family'] for reg in axon_length.index]
        family = [[family_index[v]] for v in family_region]
        family_region = [[r] for r in family_region]
        regions = list(axon_length.index)
        regions_map = ff.create_annotated_heatmap(z=family,
                                                  y=regions,
                                                  customdata=family_region,
                                                  hovertemplate=
                                                  '<b>region</b>: %{y}' +
                                                  '<br><b>family</b>: %{customdata}<extra></extra>',
                                                  colorscale=family_colors,
                                                  ygap=1,
                                                  annotation_text=[[r] for r in regions]
                                                  )

        family_unique = list(set(row_family.family))
        family_index_unique = [family_index[v] for v in family_unique]
        region_legend = ff.create_annotated_heatmap(z=[[r] for r in family_index_unique],
                                                    y=family_unique,
                                                    hovertemplate=
                                                    '<b>region</b>: %{y}<extra></extra>',
                                                    colorscale=family_colors,
                                                    ygap=4,
                                                    annotation_text=[[r] for r in family_unique]
                                                    )

        legend_fig = go.Figure(region_legend)
        legend_fig.update_layout(xaxis_showticklabels=False, yaxis_showticklabels=False)
        legend_anno = legend_fig.layout.annotations
    else:
        family_colors = distinctipy.get_colors(len(set(axon_length.index)), pastel_factor=0.7)
        family_colors = ['rgb(' + str(int(c[0] * 255)) + ',' + str(int(c[1] * 255)) + ',' + str(int(c[2] * 255)) + ')'
                         for c in family_colors]
        family_range = np.linspace(0, 1, len(set(axon_length.index)))
        family_colors = list(map(list, zip(family_range, family_colors)))
        family_index = dict(zip(list(set(axon_length.index)), list(range(len(set(axon_length.index))))))
        family = [[family_index[v]] for v in axon_length.index]
        regions = list(axon_length.index)
        regions_map = ff.create_annotated_heatmap(z=family,
                                                  y=regions,
                                                  hovertemplate=
                                                  '<b>region</b>: %{y}',
                                                  colorscale=family_colors,
                                                  ygap=1,
                                                  annotation_text=[[r] for r in regions]
                                                  )


    family_fig = go.Figure(regions_map)
    family_fig.update_layout(xaxis_showticklabels=False, yaxis_showticklabels=False)
    anno_regions = family_fig.layout.annotations

    fig = make_subplots(rows=3, cols=2, vertical_spacing=0.05, horizontal_spacing=0.05)
    for data in den_fig['data']:
        fig.add_trace(data, row=1, col=2)

    for data in labels_fig['data']:
        fig.add_trace(data, row=2, col=2)

    for data in family_fig['data']:
        fig.add_trace(data, row=3, col=1)



    heatmap_x = [neu_names] * len(regions)
    heatmap_y = np.transpose([regions] * len(neu_names))
    fig.add_trace(go.Heatmap(
        x=den_fig['layout']['xaxis']['tickvals'],
        y=family_fig['layout']['yaxis']['tickvals'],
        customdata=np.moveaxis([heatmap_x, heatmap_y], 0, -1),
        hovertemplate=
        '<b>neuron</b>: %{customdata[0]}' +
        '<br><b>region</b>: %{customdata[1]}' +
        '<br><b>value</b>: %{z}<extra></extra>',
        z=axon_length,
        colorscale='rdylbu_r',
        opacity=0.9,
        xgap=0, ygap=1,
        colorbar=dict(len=0.3, y=0.655)), row=3, col=2)

    for anno in anno_regions:
        fig.add_annotation(
            row=3,
            col=1,
            font=anno.font,
            showarrow=False,
            text=anno.text,
            x=anno.x,
            y=anno.y
        )
    if template == Template.allen:
        for data in legend_fig['data']:
            fig.add_trace(data, row=1, col=1)
        for anno in legend_anno:
            fig.add_annotation(
                row=1,
                col=1,
                font=anno.font,
                showarrow=False,
                text=anno.text,
                x=anno.x,
                y=anno.y
            )
    fig.update_layout(height=900,
        # height=800, width=800,
                      hovermode='closest',
                      template='none',
                      showlegend=False)
    fig.update_layout(xaxis={'domain': [0., 0.075],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'showticklabels': False,
                             'ticks': ""},
                      xaxis2={'domain': [0.085, 1],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""},
                      xaxis3={'domain': [0., 0.075],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""},
                      xaxis4={'domain': [0.085, 1],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""},
                      xaxis5={'domain': [0., 0.075],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': "",
                              },
                      xaxis6={'domain': [0.085, 1],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""},
                      )
    fig.update_layout(yaxis={'domain': [0.825, 1],
                             'mirror': False,
                             'showgrid': False,
                             'showline': False,
                             'zeroline': False,
                             'showticklabels': False,
                             'ticks': ""},
                      yaxis2={'domain': [0.825, 1],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""},
                      yaxis3={'domain': [0.81, 0.83],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""},
                      yaxis4={'domain': [0.81, 0.83],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""},
                      yaxis5={'domain': [0., 0.8],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""},
                      yaxis6={'domain': [0., 0.8],
                              'mirror': False,
                              'showgrid': False,
                              'showline': False,
                              'zeroline': False,
                              'showticklabels': False,
                              'ticks': ""},)
    return py.offline.plot(fig, include_plotlyjs=False, output_type='div', config={'displaylogo': False})


