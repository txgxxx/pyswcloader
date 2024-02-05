import numpy as np
import plotly as py
from seaborn._statistics import KDE
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from reader.brain import find_region, Template
from reader.io import STL_ACRO_DICT


def get_web_neuron_summary_info(neuron_num, not_valid):
    if not_valid > 0:
        succ_alert = '<div class="alert alert-success" role="alert">' \
                     'There are %d neurons in total, and all of them are valid. </div>' % neuron_num
        return succ_alert
    warn_alert = '<div class="alert alert-warning" role="alert">' \
    'There are %d neurons in total, %d of which are invalid. We have fixed them. </div>'% (neuron_num, not_valid)
    return warn_alert


def get_web_neuron_region_info(soma_info, template, annotation, resolution):
    soma_info['region'] = soma_info.apply(lambda x: find_region(x[['x', 'y', 'z']], annotation, resolution), axis=1)
    if template == Template.allen:
        soma_info['region'] = soma_info['region'].map(STL_ACRO_DICT)
    reg, num = np.unique(soma_info['region'], return_counts=True)
    reg_dict = dict(zip(reg, num))
    soma_info['count'] = [reg_dict[reg] for reg in soma_info['region']]
    soma_group = soma_info.groupby(['region', 'count', 'neuron']).count()
    classes = 'class="table table-border"'
    soma_html = soma_group.to_html(header=False)
    soma_html = soma_html.replace('class="dataframe"', classes)
    # bootstrap_link = '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">'
    soma_html = soma_html
    return soma_html


def get_web_soma_distribution(soma_info):
    x_v, x_cnt = np.unique(soma_info['x'], return_counts=True)
    kde = KDE(bw_method="scott")
    density1, support1 = kde(x1=x_v, weights=x_cnt)
    y_v, y_cnt = np.unique(soma_info['y'], return_counts=True)
    kde = KDE(bw_method="scott")
    density2, support2 = kde(x1=y_v, weights=y_cnt)
    z_v, z_cnt = np.unique(soma_info['z'], return_counts=True)
    kde = KDE(bw_method="scott")
    density3, support3 = kde(x1=z_v, weights=z_cnt)
    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        specs=[[{"type": "scatter"}],
                               [{"type": "scatter"}],
                               [{"type": "scatter"}]],
                        subplot_titles=['Anterior - Posterior', 'Dorsal - Ventral', 'Medial - Lateral', ])
    fig.add_trace(
        go.Scatter(
            x=support1,
            y=density1,
            mode='lines',
            line=dict(color='rgb(93, 164, 214)'),
            name='Anterior - Posterior',
            fill='tozeroy',
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=support2,
            y=density2,
            mode='lines',
            line=dict(color='rgb(255, 144, 14)'),
            name='Dorsal - Ventral',
            fill='tozeroy',
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=support3,
            y=density3,
            line=dict(color='rgb(44, 160, 101)'),
            mode='lines',
            name='Medial - Lateral',
            fill='tozeroy',
        ),
        row=3, col=1
    )
    fig.update_layout(
        height=600, width=1000,
        showlegend=False,)
    return py.offline.plot(fig, include_plotlyjs=False, output_type='div')

