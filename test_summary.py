import vispy
vispy.use("osmesa")
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import pandas as pd

from pyswcloader.projection import projection_batch, projection_neuron
from pyswcloader.web_summary import build_web_summary
from pyswcloader.cluster import *
from pyswcloader.reader import brain, io, swc
from pyswcloader.summary import Summary

data_path = '/home/share/cdc_algorithm/pyswcloader/manuscript/data/PFC'
axon_length = pd.read_csv('/home/dongzhou/data/mouse/scores/PFC_cluster_result/axon_length.csv', index_col=0)
scores = pd.read_csv('/home/dongzhou/data/mouse/scores/PFC_cluster_result/scores.csv', index_col=0)
n_clusters = 64
template = brain.Template.allen
save_path = '/home/dongzhou/data/mouse/scores/PFC_cluster_result/scores'
cluster_info = cluster(n_clusters,
                       template=template,
                       feature=Feature.precomputed,
                       method=Method.hierarchy,
                       matrix=scores,
                       projection=axon_length,
                       data_path=data_path,
                       save=True,
                       save_path=save_path)
plot_cluster(cluster_info, show=False, save_path=save_path, region_path=io.ALLEN_ROOT_PATH, region_opacity=0.2, bgcolor='white')


def check_data(data_path, cores):
    path_list = swc.read_neuron_path(data_path)
    results = []
    with ThreadPoolExecutor(max_workers=cores) as executor:
        results = executor.map(swc.check_swc, path_list)
    wrong_swc = len(path_list) - sum(results)
    return len(path_list), sum(results), wrong_swc
neuron_num, is_valid, not_valid = check_data(data_path, cpu_count()//2)
soma_info, _ = swc.plot_soma_distribution(data_path, save=True, save_path=save_path)

topographic_info = projection_batch.compute_projection_parallel(projection_neuron.topographic_projection_info,
                                                                data_path,
                                                                cores=cpu_count()//2,
                                                                template=template,
                                                                annotation=io.ALLEN_ANNOTATION,
                                                                resolution=10,
                                                                save=False)
show_data, _ = projection_vis.plot_topographic_projection(topographic_info,
                                                       template,
                                                       threshold=10,
                                                       p_threshold=0.05,
                                                       save=True,
                                                       save_path=save_path)


build_web_summary([neuron_num, is_valid, not_valid],
                          soma_info, cluster_info,
                          show_data, template, save_path)