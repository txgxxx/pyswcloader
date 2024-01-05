from multiprocessing import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import fastjsonschema.ref_resolver
from tqdm import tqdm, trange
import os, sys
import glob
from functools import partial
import pandas as pd
from pyswcloader.reader import *
import projection



def compute_unix_parallel(func, data_path, cores=int(cpu_count() / 2), **params):
    path_list = swc.read_neuron_path(data_path)
    data = pd.DataFrame()
    with ProcessPoolExecutor(max_workers=cores) as executor:
        data = pd.concat(list(tqdm(executor.map(partial(func, **params), path_list), total=len(path_list))))
    return data
    # pool = Pool(cores)
    # data = pd.concat(pool.map(
    #     partial(func, **params), tqdm(path_list)
    # ))
    # pool.close()
    # pool.join()
    # return data


def projection_length(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                      save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    length = pd.DataFrame()
    pool = Pool(cores)
    length = pd.concat(pool.map(
        partial(projection.projection_length, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return length


def projection_length_ipsi(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                           save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    length = pd.DataFrame()
    pool = Pool(cores)
    length = pd.concat(pool.map(
        partial(projection.projection_length_ipsi, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return length


def projection_length_contra(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                             save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    length = pd.DataFrame()
    pool = Pool(cores)
    length = pd.concat(pool.map(
        partial(projection.projection_length_contra, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return length


def _terminal_info(path, save_state, save_path):
    info = terminal_info(path, save=save_state, save_path=save_path)
    info['neuron'] = os.path.basename(path).split('.')[0]
    return info


def terminal_info(data_path, cores=int(cpu_count() / 2), save_state=False, save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    info = pd.DataFrame()
    pool = Pool(cores)
    info = pd.concat(pool.map(partial(projection.terminal_info, save_state=save_state, save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return info


def terminal_count(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                   save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    count = pd.DataFrame()
    pool = Pool(cores)
    count = pd.concat(pool.map(
        partial(projection.terminal_count, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return count


def terminal_count_ipsi(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                        save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    count = pd.DataFrame()
    pool = Pool(cores)
    count = pd.concat(pool.map(
        partial(projection.terminal_count_ipsi, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return count


def terminal_count_contra(data_path, annotation, resolution, cores=int(cpu_count() / 2), save_state=False,
                          save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    count = pd.DataFrame()
    pool = Pool(cores)
    count = pd.concat(pool.map(
        partial(projection.terminal_count_contra, annotation=annotation, resolution=resolution, save=save_state,
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
        partial(projection.topographic_projection_info, annotation=annotation, resolution=resolution, save=save_state,
                save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return topographic_info

if __name__ == '__main__':
    import time
    t1 = time.time()
    test_path = '/home/cdc/data/mouse_data/test'
    annotation = brain.read_nrrd('/home/cdc/Documents/mouse_neuron_analysis/database/annotation_10.nrrd')
    # data = topographic_projection_info_batch(test_path, annotation, 10)
    kwargs_dict = {'annotation': annotation, 'resolution': 10, 'save': False, 'save_path': os.getcwd()}
    data = compute_unix_parallel(projection.topographic_projection_info, test_path,
                                 annotation=annotation, resolution=10, save=False,
                                 save_path=os.getcwd())
    print(data.shape)
    print(data)
    print(time.time() - t1)