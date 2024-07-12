import platform
from multiprocessing import cpu_count
from multiprocessing.pool import Pool, ThreadPool
from tqdm import tqdm
from functools import partial
import pandas as pd
from pyswcloader.projection.projection_neuron import *
from pyswcloader.reader import swc, brain, io




def compute_projection_parallel(func, data_path, cores=None, **params):
    path_list = swc.read_neuron_path(data_path)

    if platform.system() == 'Linux':
        pool = Pool(cores)
    else:
        pool = ThreadPool(cores)
    data = pd.concat(pool.map(partial(func, **params), tqdm(path_list)))
    pool.close()
    pool.join()
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





