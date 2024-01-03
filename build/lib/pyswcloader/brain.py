import nrrd
import numpy as np
import pandas as pd
import json
from treelib import Tree

def read_nrrd(path):
    anno, _ = nrrd.read(path)
    print(anno.shape)
    return anno

def find_region(coords, annotataion, resolution):
    return annotataion[int(coords[0]/resolution), int(coords[1]/resolution), int(coords[2]/resolution)]

def find_unique_regions(annotation):
    region_list = list(np.unique(annotation))
    # if 0 in region_list:
    #     region_list.remove(0)
    return region_list

def _recover_tree(json_file, tree=Tree()):
    for node in json_file:
        if isinstance(node, dict):
            if node['parent_structure_id']==None:
                tree.create_node(identifier=node['id'], data={'acronym':node['acronym'], 'name':node['name']})
            else:
                tree.create_node(identifier=node['id'], data={'acronym':node['acronym'], 'name':node['name']}, 
                                 parent=node['parent_structure_id'])
            for item in node.keys():
                if isinstance(node[item], list) and len(node[item])>0:
                    _recover_tree(node[item], tree=tree)
    return tree

def allen_brain_tree(json_path):
    info = json.load(open(json_path, 'rb'))['msg']
    return _recover_tree(info, tree=Tree())

def acronym_dict(json_path):
    stl_acro_dict = {}
    tree = allen_brain_tree(json_path)
    for node in tree.nodes.keys():
        stl_acro_dict[node] = tree.nodes[node].data['acronym']
    return stl_acro_dict

def find_name_by_id(node_id, json_path):
    tree = allen_brain_tree(json_path)
    return tree.nodes[node_id].data['name']

def find_acronym_by_id(node_id, json_path):
    tree = allen_brain_tree(json_path)
    return tree.nodes[node_id].data['acronym']

def find_id_by_acronym(acro, json_path):
    tree = allen_brain_tree(json_path)
    for node in tree.nodes.keys():
        if tree.nodes[node].data['acronym'] == acro:
            return node
    return ValueError('Invalid acronym.')

def find_parent(key, layer, json_path):
    tree = allen_brain_tree(json_path)
    if isinstance(key, int):
        id = key
    elif isinstance(key, str):
        id = find_id_by_acronym(key, json_path)
    while tree.depth(id)>=layer:
        id = tree.parent(id).identifier
    if isinstance(key, int):
        return id
    elif isinstance(key, str):
        return find_acronym_by_id(id, json_path)