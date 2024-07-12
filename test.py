
import vispy
import os
# os.environ['EGL_LIBRARY'] = '/home/dongzhou/anaconda3/envs/pyswcloader/x86_64-conda-linux-gnu/sysroot/usr/lib/libEGL.so.1'

# vispy.use("osmesa")

from pyswcloader.web_summary import build_web_summary
from pyswcloader.reader import swc, io, brain
from pyswcloader import cluster
from pyswcloader.projection import *
from pyswcloader.visualization import projection_vis
import pandas as pd
import math
from multiprocessing.pool import Pool, ThreadPool


# def test_projection_length(data_path, annotation, resolution):
#     region_list = brain.find_unique_regions(annotation)
#     neuron_name = data_path.split('/')[-1].split('.')[0]
#     length = pd.DataFrame(index=[neuron_name], columns=region_list)
#     length = length.fillna(0)
#     data = swc.swc_preprocess(data_path)
#     data['region'] = data.apply(lambda x: brain.find_region(x[['x', 'y', 'z']], annotation, resolution), axis=1)
#     try:
#         for idx in data.index[1:]:
#             reg = data.loc[idx, 'region']
#             parent_idx = data.loc[idx, 'parent']
#             parent_reg = data.loc[parent_idx, 'region']
#             _is_axon = data.loc[idx, 'type']
#             _parent_is_axon = data.loc[parent_idx, 'type']
#             # if _is_axon not in [3, 4] and _parent_is_axon not in [3, 4]:
#             #     if reg == parent_reg:
#             #         length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])
#             #     else:
#             #         length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z']) / 2
#             #         length.loc[neuron_name, parent_reg] += math.dist(data.loc[idx, 'x':'z'],
#             #                                                          data.loc[parent_idx, 'x':'z']) / 2
#     except:
#         print(data_path)
#         return data_path
#     else:
#         return ''
#
#
# if __name__ == '__main__':
#     data_path = '/home/dongzhou/data/mouse/PFC_20240527'
#     opts = []
#     for neuron in os.listdir(data_path):
#         if neuron.endswith('.swc'):
#             opts.append((os.path.join(data_path, neuron), io.ALLEN_ANNOTATION, 10))
#     pool = Pool(32)
#     res = pool.starmap(test_projection_length, opts)
#     pool.close()
#     pool.join()
#     txt = '/home/dongzhou/data/mouse/PFC_wrong_file.txt'
#     with open(txt, 'w') as f:
#         for i in res:
#             if i:
#                 f.write(i + '\n')


data = pd.read_csv('/home/dongzhou/data/mouse/test_res/cluster_results.csv')
print(data)
print(io.ALLEN_ROOT_PATH)
cluster.plot_cluster(data, show=False, save_path='/home/dongzhou/data/mouse/test_res/clusters', region_path=io.ALLEN_ROOT_PATH, region_opacity=0.1)

# from xvfbwrapper import Xvfb
#
# vdisplay = Xvfb()
# vdisplay.start()
#
# try:
#     pass
#     # launch stuff inside virtual display here.
# finally:
#     # always either wrap your usage of Xvfb() with try / finally,
#     # or alternatively use Xvfb as a context manager.
#     # If you don't, you'll probably end up with a bunch of junk in /tmp
#     vdisplay.stop()
# pfc_path = '/home/dongzhou/data/mouse/pfc20230717'
# length_path = '/home/dongzhou/data/mouse/proj_info/pfc/length.csv'
# scores_path = '/home/dongzhou/data/mouse/proj_info/pfc/min_score.csv'
# save_path = '/home/dongzhou/data/mouse/test_res'
#
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
# neurons = os.listdir(pfc_path)
# #
# length = pd.read_csv(length_path, index_col=0)
# # print(length)
# # print(io.STL_ACRO_DICT)
# length.columns = list(map(int, length.columns))
# length.columns = length.columns.map(io.STL_ACRO_DICT)
#
# cluster_info = pd.read_csv('/home/dongzhou/data/mouse/test_res/cluster_results.csv', index_col=0)
# # print(length)
# #
# scores = pd.read_csv(scores_path, index_col=0)
# scores.index = [os.path.join(pfc_path, neu+'_reg.swc') for neu in scores.index]
# scores.columns = [os.path.join(pfc_path, neu+'_reg.swc') for neu in scores.columns]
# #
# soma_info = swc.plot_soma_distribution(pfc_path, save=False)
# # cluster_info = cluster.cluster(n_cluster=64, feature=cluster.Feature.precomputed, projection=length,matrix=scores, data_path=pfc_path, save_path=save_path, save=True)
# # cluster.plot_cluster(cluster_info, show=False, save_path=save_path, region_path='/home/dongzhou/data/mouse/997.obj', region_opacity=0.2, bgcolor='white')
# topographic_info = projection_batch.compute_projection_parallel(projection_neuron.topographic_projection_info,
#                                                                 pfc_path,
#                                                                 cores=64,
#                                                                 template=brain.Template.allen,
#                                                                 annotation=io.ALLEN_ANNOTATION,
#                                                                 resolution=10,
#                                                                 save=False)
#
# show_data = projection_vis.plot_topographic_projection(topographic_info,
#                                                                brain.Template.allen,
#                                                                threshold=10,
#                                                                p_threshold=0.05,
#                                                                save=True,
#                                                                save_path=save_path)
#
# build_web_summary([len(neurons), len(neurons), 0],
#                           soma_info, cluster_info,
#                           show_data, brain.Template.allen, save_path, scores)