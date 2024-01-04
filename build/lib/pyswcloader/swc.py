import pandas as pd
import numpy as np
import os, sys
from tqdm import tqdm
from treelib import Tree
import math
import random
import copy
import seaborn as sns
from matplotlib import pyplot as plt
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def read_swc(path):
    data = pd.read_csv(path, sep=' ', header=None, comment='#')
    data.columns = ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']
    data.index = np.arange(1, len(data)+1)
    return data

def swc_preprocess(path, save_path=None, save=False, check_validity=True, flip=True, dimension=[13200, 8000, 11400]):
    data = read_swc(path)

    if flip==True and float(data.loc[1, 'z'])>(11400/2):
        data.z = dimension[2] - data.z

    if check_validity==True:
        if len(data.loc[data.parent==-1])>1:
            print(path + ': more than one soma detected -> parent=-1.')
            child_list = list(data.loc[data.parent==-1].index[1:])
            while len(child_list)>0:
                data = data.loc[~data.index.isin(child_list), :]
                child_list = list(data[data.parent.isin(child_list)].index)

        elif len(data.loc[data.type==1])>1:
            print(path + ': more than one soma detected -> type=1.')
            data.loc[data[data.type==1].index[1:],'type'] = 0

    if data.x.max()>dimension[0]:
        data.x = [item if item<dimension[0] else dimension[0]-1 for item in data.x]
        print('X axis exceeds boundary.')
    if data.y.max()>dimension[1]:
        data.y = [item if item<dimension[1] else dimension[1]-1 for item in data.y]
        print('Y axis exceeds boundary.')
    if data.z.max()>dimension[2]:
        data.z = [item if item<dimension[2] else dimension[2]-1 for item in data.z]
        print('Z axis exceeds boundary.')    

    if save==True:
        data.to_csv(save_path, sep=" ", header=None, index=None)
    return data

def swc_tree(path):
    tree = Tree()
    data = read_swc(path)
    for _, row in data.iterrows():
        tree.create_node(
            tag=int(row.id),
            identifier=int(row.id),
            data=row.loc[['x', 'y', 'z']].to_dict(),
            parent=row.parent if row.parent != -1 else None
        )
    return tree

def total_length(path):
    length = 0
    data = read_swc(path)
    for idx in data.index[1:]:
        parent_idx = data.loc[idx, 'parent']
        _is_axon = data.loc[idx, 'type']
        _parent_is_axon = data.loc[parent_idx, 'type']
        if _is_axon not in [3, 4] and _parent_is_axon not in [3, 4]:
            length += math.dist(data.loc[idx, 'x':'z'], data.loc[parent_idx, 'x':'z'])
    return length

def find_longest_stem(path):
    tree = swc_tree(path)
    weight_dict = {}
    for node in list(tree.nodes.keys())[1:]:
        weight_dict[node] = math.dist(tree.nodes[node].data.values(), tree.parent(node).data.values())
    paths = tree.paths_to_leaves()
    stem_list = paths[0].copy()
    mark = np.sum([weight_dict[item] for item in stem_list[1:]])
    for _path in paths:
        path_length = np.sum([weight_dict[item] for item in _path[1:]])
        if path_length > mark:
            stem_list = _path.copy()
            mark = path_length
    data = read_swc(path)
    return data.loc[stem_list, :]

def find_skeleton_tree(path):
    tree = swc_tree(path=path)
    # node_list = list(tree.nodes.keys())
    root = tree.root
    skeleton_tree = copy.deepcopy(tree)
    for node in tree.nodes.keys():
        if node != root and len(tree.children(node)) == 1:
            skeleton_tree.link_past_node(node)
    return skeleton_tree

def find_path_to_parent(node_id, parent_id, tree):
    par = node_id
    path = [node_id]
    while par != parent_id:
        par = tree.parent(par).identifier
        path.append(par)
    path.reverse()
    return path

def downsample(path, eps=1, save_path=None):
    from rdp import rdp
    tree = swc_tree(path)
    data = read_swc(path)
    skeleton_tree = find_skeleton_tree(path)
    new_data = pd.DataFrame(columns=data.columns)
    # add root
    ref_dict = {}
    ref_dict[data.iloc[0, 0]] = 1
    new_data = new_data.append(data.iloc[0,:])

    for node in list(skeleton_tree.nodes.keys())[1:]:
        parent = skeleton_tree.parent(node).tag
        # add parent
        if parent not in list(ref_dict.keys()):
            ref_dict[parent] = new_data.id.max()+1
        if ref_dict[parent] not in list(new_data.id):
            new_data = new_data.append({
                'id':ref_dict[parent],
                'x':data.loc[parent, 'x'],
                'y':data.loc[parent, 'y'],
                'z':data.loc[parent, 'z'],
                'radius': data.loc[parent, 'radius'],
                'parent': data.loc[parent, 'parent']
                }, ignore_index=True)

        # downsample branch points
        branch_path = find_path_to_parent(node_id=node, parent_id=parent, tree=tree)
        bp = data.loc[branch_path[1:-1], :]
        rdp_coords = rdp(bp[['x', 'y', 'z']], epsilon=eps)
        if len(rdp_coords) > 0:
            new_bp = pd.DataFrame(columns=data.columns)
            new_bp[['x','y','z']] = rdp_coords
            new_bp.id = np.arange(new_data.id.max()+1, new_data.id.max()+1+len(rdp_coords), 1)
            new_bp.iloc[1:, 6] = list(new_bp.id)[:-1]
            new_bp.iloc[0, 6] = ref_dict[parent] # parent
        
        # add end node
            new_bp = new_bp.append(data.loc[node, :])
            # end node id
            new_bp.iloc[-1, 0] = new_bp.iloc[-2, 0]+1 
            ref_dict[node] = new_bp.iloc[-1, 0]
            # edn node parent
            new_bp.iloc[-1, 6] = new_bp.iloc[-2, 0]

            new_data = pd.concat([new_data, new_bp], axis=0)
        else:
            ref_dict[node] = new_data.id.max()+1
            new_data = new_data.append({
                'id': ref_dict[node],
                'type': data.loc[node, 'type'],
                'x': data.loc[node, 'x'],
                'y': data.loc[node, 'y'],
                'z': data.loc[node, 'z'],
                'radius': data.loc[node, 'radius'],
                'parent': ref_dict[parent]
                }, ignore_index=True)
    new_data = new_data.fillna(0)
    new_data.index = list(new_data.id)

    if save_path != None:
        new_data.to_csv(save_path, sep=" ", header=None, index=None)
    return new_data

def read_neuron_path(data_path):
    path_list = glob.glob(os.path.join(data_path, '**/*.swc'), recursive=True)
    return path_list

def plot_soma_distribution(data_path, **kwargs):
    path_list = read_neuron_path(data_path)
    soma_info = []
    for path in tqdm(path_list):
        data = read_swc(path)
        soma_info.append(np.array(data.loc[1, ['x', 'y', 'z']]))
    soma_info = pd.DataFrame(soma_info)
    soma_info.columns = ['x', 'y', 'z']
    fig, _ = plt.subplots(nrows=3, sharex=False, sharey=False)
    cnt = 1
    for axis in ['x', 'y', 'z']:
        plt.subplot(310+cnt)
        sns.kdeplot(list(soma_info[axis]), **kwargs)
        plt.ylabel('')
        ax = plt.gca()
        if axis == 'x':
            ax.set_title('Anterior - Posterior')
        elif axis == 'y':
            ax.set_title('Dorsal - Ventral')
        else:
            ax.set_title('Lateral - Medial')
        cnt += 1
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.subplots_adjust(hspace=1)
    plt.subplots_adjust(wspace=0)
    return fig
    
