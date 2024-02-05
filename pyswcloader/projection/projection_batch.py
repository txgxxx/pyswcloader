import os
import platform
from multiprocessing import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from functools import partial
import pandas as pd
from .projection_neuron import *


def compute_projection_parallel(func, data_path, cores=int(cpu_count() / 2), **params):
    path_list = swc.read_neuron_path(data_path)
    if platform.system() == 'Linux':
        with ProcessPoolExecutor(max_workers=cores) as executor:
            results = list(tqdm(executor.map(partial(func, **params), path_list), total=len(path_list)))
    else:
        with ThreadPoolExecutor(max_workers=cores) as executor:
            results = list(tqdm(executor.map(partial(func, **params), path_list), total=len(path_list)))
    data = pd.concat(results)
    return data


def projection_length_batch(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                            save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    length = pd.DataFrame()
    pool = Pool(cores)
    length = pd.concat(pool.map(
        partial(projection_length, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return length


def projection_length_ipsi_batch(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                                 save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    length = pd.DataFrame()
    pool = Pool(cores)
    length = pd.concat(pool.map(
        partial(projection_length_ipsi, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return length


def projection_length_contra_batch(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                                   save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    length = pd.DataFrame()
    pool = Pool(cores)
    length = pd.concat(pool.map(
        partial(projection_length_contra, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return length


def terminal_info_batch(data_path, cores=int(cpu_count() / 2), save_state=False, save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    info = pd.DataFrame()
    pool = Pool(cores)
    info = pd.concat(pool.map(partial(terminal_info, save_state=save_state, save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return info


def terminal_count_batch(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                         save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    count = pd.DataFrame()
    pool = Pool(cores)
    count = pd.concat(pool.map(
        partial(terminal_count, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return count


def terminal_count_ipsi_batch(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                              save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    count = pd.DataFrame()
    pool = Pool(cores)
    count = pd.concat(pool.map(
        partial(terminal_count_ipsi, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return count


def terminal_count_contra_batch(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                                save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    count = pd.DataFrame()
    pool = Pool(cores)
    count = pd.concat(pool.map(
        partial(terminal_count_contra, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return count


def topographic_projection_info_batch(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                                      save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    topographic_info = pd.DataFrame()
    pool = Pool(cores)
    topographic_info = pd.concat(pool.map(
        partial(topographic_projection_info, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return topographic_info


# if __name__ == '__main__':
#     import time
#
#     t1 = time.time()
#     test_path = '/home/cdc/data/mouse_data/test'
#     annotation = io.read_nrrd('/home/cdc/Documents/mouse_neuron_analysis/database/annotation_10.nrrd')
#     # data = topographic_projection_info_batch(test_path, annotation, 10)
#     kwargs_dict = {'annotation': annotation, 'resolution': 10, 'save': False, 'save_path': os.getcwd()}
#     data = compute_projection_parallel(topographic_projection_info, test_path, cores=4, template=brain.Template.allen,
#                                        annotation=annotation, resolution=10, save=False, save_path=None)
#     print(data.shape)
#     print(data)
#     data.to_csv('/home/cdc/Downloads/topo.csv')
#     print(time.time() - t1)
