import numpy as np
import plotly as py
from seaborn._statistics import KDE
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_web_neuron_summary_info(neuron_num, not_valid):
    if not_valid == 0:
        succ_alert = '<div class="alert alert-success" role="alert">' \
                     'There are %d neurons in total, and all of them are valid. </div>' % neuron_num
        return succ_alert
    warn_alert = '<div class="alert alert-warning" role="alert">' \
    'There are %d neurons in total, %d of which are invalid. We have fixed them. </div>'% (neuron_num, not_valid)
    return warn_alert


def get_web_neuron_region_info(projection):
    projection['axon_length'] = [np.mean(projection.loc[reg]) for reg in projection.index]
    projection = projection.sort_values(by='axon_length', ascending=False)

    fig = go.Figure(
        go.Bar(
            x=projection.index,
            y=projection.axon_length,
            # customdata=list(topo_info.p_value),
            # marker_color='black',
            opacity=0.7,
            width=0.25,
            hovertemplate=
            '<b>region</b>: %{x}' +
            '<br><b>mean axon length</b>: %{y:.3f}<extra></extra>'

        )
    )
    fig.update_layout(
        height=300,
        # width=800,
        showlegend=False, )
    return py.offline.plot(fig, include_plotlyjs=False, output_type='div', config= {'displaylogo': False})


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
            name='Lateral - Medial',
            fill='tozeroy',
        ),
        row=3, col=1
    )
    fig.update_annotations(font_size=12)
    fig.update_layout(
        height=450,
        # width=800,
        showlegend=False,)
    return py.offline.plot(fig, include_plotlyjs=False, output_type='div', config= {'displaylogo': False})

