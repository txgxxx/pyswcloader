from multiprocessing import cpu_count, Pool
from tqdm import tqdm, trange
import os, sys
import glob
from functools import partial
import pandas as pd

from . import projection
from . import swc

def projection_length(data_path, annotation, resolution, cores=int(cpu_count()/2), save_state=False, save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    length = pd.DataFrame()
    pool = Pool(cores)
    length = pd.concat(pool.map(partial(projection.projection_length, annotation=annotation, resolution=resolution, save=save_state, save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return length

def projection_length_ipsi(data_path, annotation, resolution, cores=int(cpu_count()/2), save_state=False, save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    length = pd.DataFrame()
    pool = Pool(cores)
    length = pd.concat(pool.map(partial(projection.projection_length_ipsi, annotation=annotation, resolution=resolution, save=save_state, save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return length

def projection_length_contra(data_path, annotation, resolution, cores=int(cpu_count()/2), save_state=False, save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    length = pd.DataFrame()
    pool = Pool(cores)
    length = pd.concat(pool.map(partial(projection.projection_length_contra, annotation=annotation, resolution=resolution, save=save_state, save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return length

def _terminal_info(path, save_state, save_path):
    info = projection.terminal_info(path, save=save_state, save_path=save_path)
    info['neuron'] = os.path.basename(path).split('.')[0]
    return info

def terminal_info(data_path, cores=int(cpu_count()/2), save_state=False, save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    info = pd.DataFrame()
    pool = Pool(cores)
    info = pd.concat(pool.map(partial(_terminal_info, save_state=save_state, save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return info

def terminal_count(data_path, annotation, resolution, cores=int(cpu_count()/2), save_state=False, save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    count = pd.DataFrame()
    pool = Pool(cores)
    count = pd.concat(pool.map(partial(projection.terminal_count, annotation=annotation, resolution=resolution, save=save_state, save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return count

def terminal_count_ipsi(data_path, annotation, resolution, cores=int(cpu_count()/2), save_state=False, save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    count = pd.DataFrame()
    pool = Pool(cores)
    count = pd.concat(pool.map(partial(projection.terminal_count_ipsi, annotation=annotation, resolution=resolution, save=save_state, save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return count

def terminal_count_contra(data_path, annotation, resolution, cores=int(cpu_count()/2), save_state=False, save_path=os.getcwd()):
    path_list = swc.read_neuron_path(data_path)
    count = pd.DataFrame()
    pool = Pool(cores)
    count = pd.concat(pool.map(partial(projection.terminal_count_contra, annotation=annotation, resolution=resolution, save=save_state, save_path=save_path), tqdm(path_list)))
    pool.close()
    pool.join()
    return count


