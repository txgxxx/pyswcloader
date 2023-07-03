import numpy as np
import pandas as pd
import os, sys
import math

from . import brain
from . import swc

def projection_length(data_path, annotation, resolution, save=False, save_path=os.getcwd()):
    region_list = brain.find_unique_regions(annotation)
    neuron_name = data_path.split('/')[-1].split('.')[0]
    length = pd.DataFrame(index=[neuron_name], columns=region_list)
    length = length.fillna(0)
    data = swc.swc_preprocess(data_path)
    data['region'] = data.apply(lambda x: brain.find_region(x[['x','y','z']], annotation, resolution), axis=1)
    for idx in data.index[1:]:
        reg = data.loc[idx, 'region']
        parent_idx = data.loc[idx, 'parent']
        parent_reg = data.loc[parent_idx, 'region']
        _is_axon = data.loc[idx, 'type']
        _parent_is_axon = data.loc[parent_idx, 'type']
        if _is_axon not in [3, 4] and _parent_is_axon not in [3, 4]:
            if reg==parent_reg:
                length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])
            else:
                length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/2
                length.loc[neuron_name, parent_reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/2
    if 0 in length.columns:
        length = length.drop(columns=0)
    if save == True:
        length.to_csv(os.path.join(save_path, neuron_name+'_projection_length.csv'))
    return length

def projection_length_ipsi(data_path, annotation, resolution, save=False, save_path=os.getcwd()):
    region_list = brain.find_unique_regions(annotation)
    neuron_name = data_path.split('/')[-1].split('.')[0]
    length = pd.DataFrame(index=[neuron_name], columns=region_list)
    length = length.fillna(0)
    data = swc.swc_preprocess(data_path)
    data['region'] = data.apply(lambda x: brain.find_region(x[['x','y','z']], annotation, resolution), axis=1)
    mid = int(annotation.shape[2]/2 * resolution)
    mark = data.loc[1, 'z']<mid
    data['ipsi'] = [(item<mid)==mark for item in data.z]
    for idx in data.index[1:]:
        reg = data.loc[idx, 'region']
        ipsi = data.loc[idx, 'ipsi']
        parent_idx = data.loc[idx, 'parent']
        parent_reg = data.loc[parent_idx, 'region']
        parent_ipsi = data.loc[parent_idx, 'ipsi']
        _is_axon = data.loc[idx, 'type']
        _parent_is_axon = data.loc[parent_idx, 'type']
        if _is_axon not in [3, 4] and _parent_is_axon not in [3, 4]:
            if ipsi==parent_ipsi and ipsi==True:
                if reg==parent_reg:
                    length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])
                else:
                    length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/2
                    length.loc[neuron_name, parent_reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/2
            elif ipsi != parent_ipsi:
                if reg==parent_reg:
                    length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/2
                else:
                    length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/4
                    length.loc[neuron_name, parent_reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/4
    if 0 in length.columns:
        length = length.drop(columns=0)
    if save == True:
        length.to_csv(os.path.join(save_path, neuron_name+'_projection_length_ipsi.csv'))
    return length

def projection_length_contra(data_path, annotation, resolution, save=False, save_path=os.getcwd()):
    region_list = brain.find_unique_regions(annotation)
    neuron_name = data_path.split('/')[-1].split('.')[0]
    length = pd.DataFrame(index=[neuron_name], columns=region_list)
    length = length.fillna(0)
    data = swc.swc_preprocess(data_path)
    data['region'] = data.apply(lambda x: brain.find_region(x[['x','y','z']], annotation, resolution), axis=1)
    mid = int(annotation.shape[2]/2 * resolution)
    mark = data.loc[1, 'z']<mid
    data['ipsi'] = [(item<mid)==mark for item in data.z]
    for idx in data.index[1:]:
        reg = data.loc[idx, 'region']
        ipsi = data.loc[idx, 'ipsi']
        parent_idx = data.loc[idx, 'parent']
        parent_reg = data.loc[parent_idx, 'region']
        parent_ipsi = data.loc[parent_idx, 'ipsi']
        _is_axon = data.loc[idx, 'type']
        _parent_is_axon = data.loc[parent_idx, 'type']
        if _is_axon not in [3, 4] and _parent_is_axon not in [3, 4]:
            if ipsi==parent_ipsi and ipsi==False:
                if reg==parent_reg:
                    length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])
                else:
                    length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/2
                    length.loc[neuron_name, parent_reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/2
            elif ipsi != parent_ipsi:
                if reg==parent_reg:
                    length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/2
                else:
                    length.loc[neuron_name, reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/4
                    length.loc[neuron_name, parent_reg] += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])/4
    if 0 in length.columns:
        length = length.drop(columns=0)
    if save == True:
        length.to_csv(os.path.join(save_path, neuron_name+'_projection_length_contra.csv'))
    return length

def terminal_info(data_path, save=False, save_path=os.getcwd()):
    neuron_name = data_path.split('/')[-1].split('.')[0]
    data = swc.swc_preprocess(data_path)
    tree = swc.swc_tree(data_path)
    terminal_list = []
    for item in tree.filter_nodes(lambda x: x.is_leaf()):
        terminal_list.append(item.identifier)
    info = data.loc[data.id.isin(terminal_list)]
    info = info[~info.type.isin([3, 4])]
    if save == True:
        info.to_csv(os.path.join(save_path, neuron_name+'_terminal_info.csv'))
    return info

def terminal_count(data_path, annotation, resolution, save=False, save_path=os.getcwd()):
    neuron_name = data_path.split('/')[-1].split('.')[0]
    info = terminal_info(data_path)
    info['region'] = info.apply(lambda x: brain.find_region(x[['x','y','z']], annotation, resolution), axis=1)
    region_list = brain.find_unique_regions(annotation)
    count = pd.DataFrame(index=[neuron_name], columns=region_list)
    count_dict = info.region.value_counts().to_dict()
    count.loc[neuron_name,:] = count.columns.map(count_dict)
    count = count.fillna(0)
    if 0 in count.columns:
        count = count.drop(columns=0)
    if save == True:
        count.to_csv(os.path.join(save_path, neuron_name+'_terminal_count.csv'))
    return count

def terminal_count_ipsi(data_path, annotation, resolution, save=False, save_path=os.getcwd()):
    neuron_name = data_path.split('/')[-1].split('.')[0]
    info = terminal_info(data_path)
    info['region'] = info.apply(lambda x: brain.find_region(x[['x','y','z']], annotation, resolution), axis=1)
    mid = int(annotation.shape[2]/2 * resolution)
    info['ipsi'] = [item<mid for item in info.z] # because somas are always mirrored to the left brain
    info = info[info.ipsi==True]
    region_list = brain.find_unique_regions(annotation)
    count = pd.DataFrame(index=[neuron_name], columns=region_list)
    count_dict = info.region.value_counts().to_dict()
    count.loc[neuron_name,:] = count.columns.map(count_dict)
    count = count.fillna(0)
    if 0 in count.columns:
        count = count.drop(columns=0)
    if save == True:
        count.to_csv(os.path.join(save_path, neuron_name+'_terminal_count_ipsi.csv'))
    return count

def terminal_count_contra(data_path, annotation, resolution, save=False, save_path=os.getcwd()):
    neuron_name = data_path.split('/')[-1].split('.')[0]
    info = terminal_info(data_path)
    info['region'] = info.apply(lambda x: brain.find_region(x[['x','y','z']], annotation, resolution), axis=1)
    mid = int(annotation.shape[2]/2 * resolution)
    info['ipsi'] = [item<mid for item in info.z] # because somas are always mirrored to the left brain
    info = info[info.ipsi==False]
    region_list = brain.find_unique_regions(annotation)
    count = pd.DataFrame(index=[neuron_name], columns=region_list)
    count_dict = info.region.value_counts().to_dict()
    count.loc[neuron_name,:] = count.columns.map(count_dict)
    count = count.fillna(0)
    if 0 in count.columns:
        count = count.drop(columns=0)
    if save == True:
        count.to_csv(os.path.join(save_path, neuron_name+'_terminal_count_contra.csv'))
    return count
