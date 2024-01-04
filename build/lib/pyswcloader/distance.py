import numpy as np
import os, sys
from sklearn.neighbors import BallTree
from statistics import mean
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import trange, tqdm
import pandas as pd
from collections import Counter

from . import swc

def _find_ratio_score(n):
    n = np.array(n)
    mean = np.mean(n)
    mean_pct = float(len(n[n<np.mean(n)])/len(n))
    maxi = np.max(n)
    return mean*mean_pct+maxi*(1-mean_pct)

def morphology_distance(neuron1_path, neuron2_path):
    n1 = swc.read_swc(neuron1_path)
    n2 = swc.read_swc(neuron2_path)
    P = n1.loc[:, 'x':'z']
    Q = n2.loc[:, 'x':'z']
    tree = BallTree(P, leaf_size=256)
    dist, _ = tree.query(Q)
    inv = [item[0] for item in np.array(sorted(dist))]
    tree = BallTree(Q, leaf_size=256)
    dist, _ = tree.query(P)
    rev = [item[0] for item in np.array(sorted(dist))]
    return mean([_find_ratio_score(inv), _find_ratio_score(rev)])

def _set_call_back(idx, pairs, save_path):
    score = morphology_distance(pairs[idx][0], pairs[idx][1])
    n1 = pairs[idx][0]
    n2 = pairs[idx][1]
    f = open(save_path, 'a+')
    save_str = n1+' '+n2+' '+str(score)+'\n'
    f.write(save_str)
    f.close()
    return score

def morphology_matrix(data_path, cores=int(cpu_count()/2), save_path=os.path.join(os.getcwd(), 'scores_record.txt')):
    path_list = swc.read_neuron_path(data_path)
    pairs = []
    for i in range(len(path_list)):
            for j in range(i, len(path_list)):
                pairs.append([path_list[i], path_list[j]])
    if os.path.exists(save_path)==True:
        exitsting = pd.read_csv(save_path, sep=' ', header=None)
        cnt_existing = exitsting[0].value_counts().to_dict()
        cnt_total = Counter(np.array(pairs)[:,0])
        done = []
        for key in cnt_existing.keys():
            if cnt_existing[key]==cnt_total[key]:
                done.append(key)
        pairs = []
        for i in range(len(path_list)):
            if path_list[i] not in done:
                for j in range(i, len(path_list)):
                    pairs.append([path_list[i], path_list[j]])
    pool = Pool(cores)
    pool.map(partial(_set_call_back, pairs=pairs, save_path=save_path), trange(len(pairs)))
    pool.close()
    pool.join()
    print('Calculation done. Aggregating data...')
    info = pd.read_csv(save_path, sep=' ', header=None)
    info = info.drop_duplicates()
    _score = info.pivot_table(index=0, columns=1, values=2)
    _score = _score.fillna(0)
    score = _score + _score.T
    score.index = list(score.index)
    score.columns = list(score.columns)
    score = score.loc[score.index.isin(path_list), score.columns.isin(path_list)]
    # score.index = [item.split('/')[-1].split('.')[0] for item in list(score.index)]
    # score.columns = [item.split('/')[-1].split('.')[0] for item in list(score.columns)]
    return score
    



