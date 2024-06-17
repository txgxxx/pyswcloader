from pyswcloader.reader.io import *
from pyswcloader.reader.brain import find_region
import numpy as np
import pandas as pd
import glob
from sklearn.neighbors import NearestNeighbors

hpf_dict = {}
for key in ['CA1', 'CA2', 'CA3', 'DG', 'ENT', 'PAR', 'POST', 'PRE', 'SUB']:
    hpf_dict[key] = [STL_ACRO_DICT[item] for item in ALLEN_BRAIN_TREE.subtree(ACRO_STL_DICT[key]).nodes.keys()]

def find_key(item):
    for key in hpf_dict.keys():
        if item in hpf_dict[key]:
            return key
    return None

skeleton_axis = {}
for lateral in ['left', 'right']:
    skeleton_axis[lateral] = {}
    for file in glob.glob(os.path.join(CURRENT_WD, 'database/skeleton', lateral, '*.csv')):
        region = os.path.basename(file).split('_')[0]
        skeleton_axis[lateral][region] = np.array(pd.read_csv(file)[::-1][['X', ' Y', ' Z']])

def find_longitudinal_axis(coords, annotation=ALLEN_ANNOTATION):
    lateral = 'left' if coords[2]<(11400/2) else 'right'
    region = find_region(coords, annotation, 10)
    if region > 0:
        region = STL_ACRO_DICT[region]
        region = find_key(region)
#         region = list(filter(lambda text: text in region, skeleton_axis[lateral].keys()))
        if region in skeleton_axis[lateral].keys():
#             region = region[0]
            data = skeleton_axis[lateral][region]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data)
            _, index = nbrs.kneighbors([coords])
            return int(index[0][0])
    return None