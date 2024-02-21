import plotly as py
import plotly.graph_objects as go

from ..reader.brain import Template


def get_web_topo_info(topo_info, template):
    fig = go.Figure(
        go.Bar(
            x=topo_info.index,
            y=topo_info.r_value,
            customdata=list(topo_info.p_value),
            # marker_color='black',
            opacity=0.7,
            width=0.25,
            hovertemplate=
            '<b>region</b>: %{x}' +
            '<br><b>rvalue</b>: %{y:.3f}' +
            '<br><b>pvalue</b>: %{customdata:.3f}<extra></extra>'

        )
    )
    if template == Template.allen:
        vlinelist = topo_info.family
        start = -0.5
        mark = vlinelist[0]
        for item in vlinelist[1:]:
            start += 1
            if item != mark:
                fig.add_vline(x=start, line_width=3, line_dash='dash', line_color='grey')
            mark = item
    fig.update_layout(height=300,
                      # width=800,
                      )
    return py.offline.plot(fig, include_plotlyjs=False, output_type='div')
