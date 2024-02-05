import os
import nrrd
import jinja2
import pandas as pd
from scipy.spatial.distance import squareform

from reader.brain import Template
from web_cluster_info import get_web_projection_info, get_web_cluster_info, get_web_neuron_plot
from web_neuron_info import get_web_neuron_summary_info, get_web_soma_distribution, get_web_neuron_region_info
from web_topo_info import get_web_topo_info

TEMPLATE_PATH = './template/template.html'

def build_web_summary(neuron_info, soma_info, cluster_info, topo_info, template, annotation, resolution, save_path):
    neuron_web_info = get_web_neuron_summary_info(neuron_info[0], neuron_info[2])
    # neuron_region_web_info = get_web_neuron_region_info(soma_info, template, annotation, resolution)
    soma_web_info = get_web_soma_distribution(soma_info)
    cluster_web_info = get_web_cluster_info(cluster_info)
    info = pd.read_csv(os.path.join(save_path, 'scores_record.txt'), sep=' ', header=None)
    info = info.drop_duplicates()
    _score = info.pivot_table(index=0, columns=1, values=2)
    _score = _score.fillna(0)
    score = _score + _score.T
    score.index = list(score.index)
    score.columns = list(score.columns)

    projection_pattern = pd.read_csv('/home/cdc/data/mouse_data/temp/projection_pattern.csv', index_col=0)
    projection_web_info = get_web_projection_info(cluster_info, score, projection_pattern, template)
    img_web_html = get_web_neuron_plot(save_path)
    topo_web_info = get_web_topo_info(topo_info, template)
    env = jinja2.environment.Environment(loader=jinja2.FileSystemLoader(searchpath=''))
    template_html = env.get_template(TEMPLATE_PATH)
    html = template_html.render(neuron_info_summary=neuron_web_info,
                                # neuron_region_table=neuron_region_web_info,
                                soma_distribution_plot=soma_web_info,
                                cluster_info_table=cluster_web_info,
                                projection_pattern_plot=projection_web_info,
                                neuron_plot=img_web_html,
                                topo_plot=topo_web_info
                                )
    with open('Report.html', 'w') as f:
        f.write(html)


if __name__ == '__main__':
    soma_info = pd.read_csv('/home/cdc/data/mouse_data/temp/soma_info.csv')
    cluster_info = pd.read_csv('/home/cdc/data/mouse_data/temp/cluster_results.csv', index_col=0)
    # print(cluster_info)
    topo_info = pd.read_csv('/home/cdc/Documents/VisualiaztionTest/neuron_cluster_vis/topo_info.csv')
    anno_path = '/home/cdc/Documents/pyswcloader/pyswcloader/database/annotation_10.nrrd'
    anno, _ = nrrd.read(anno_path)
    build_web_summary([37, 37, 0], soma_info, cluster_info, topo_info,
                      template=Template.allen, annotation=anno, resolution=10,
                      save_path='/home/cdc/data/mouse_data/temp/')





