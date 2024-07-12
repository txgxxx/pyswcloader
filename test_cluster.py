import os
import argparse
import time
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
from functools import partial
from tqdm import trange
from multiprocessing.pool import Pool
from collections import Counter
from pyswcloader import cluster
from pyswcloader.reader import brain, io, swc
from pyswcloader.projection import projection_neuron, projection_batch
from pyswcloader import distance


def parse_interval(s):
    return float(s.strip("([])").split(",")[-1])

def csv_to_score_fn(fpath):
    df = pd.read_csv(fpath, index_col=0)
    return table_to_score_fn(
        [parse_interval(s) for s in df.index],
        [parse_interval(s) for s in df.columns],
        df.to_numpy(),
    )


def table_to_score_fn(dist_thresholds, dot_thresholds, cells):
    dist_thresholds = list(dist_thresholds)
    dist_thresholds[-1] = np.inf
    dist_bins = np.array(dist_thresholds, float)

    dot_thresholds = list(dot_thresholds)
    dot_thresholds[-1] = np.inf
    dot_bins = np.array(dot_thresholds, float)

    def fn(dist, dot):
        return cells[
            np.digitize(dist, dist_bins),
            np.digitize(dot, dot_bins),
        ]

    return fn

def calc_self_hit(cn):
    return cn.shape[0] * csv_to_score_fn('smat_fcwb.csv')(0, 1)

def calculate_dist(query_path, target_path):

    n1 = swc.read_swc(query_path)
    n2 = swc.read_swc(target_path)
    Q = n1.loc[:, 'x':'z']
    T = n2.loc[:, 'x':'z']
    tree = BallTree(Q, leaf_size=256)
    dist1, _ = tree.query(T, k=1)

    tree = BallTree(T, leaf_size=256)
    dist2, _ = tree.query(Q, k=1)
    scores1 = csv_to_score_fn('smat_fcwb.csv')(dist1, 1).sum() / calc_self_hit(n1)
    scores2 = csv_to_score_fn('smat_fcwb.csv')(dist2, 1).sum() / calc_self_hit(n2)
    return scores1, scores2

def _set_call_back(idx, pairs, save_path):
    score = calculate_dist(pairs[idx][0], pairs[idx][1])
    n1 = pairs[idx][0]
    n2 = pairs[idx][1]
    f = open(save_path, 'a+')
    save_str = n1 + ' ' + n2 + ' ' + str(score[0]) + ' ' + str(score[1]) + '\n'
    f.write(save_str)
    f.close()
    return score

def nblast(data_path, cores, save_path):

    path_list = swc.read_neuron_path(data_path)
    pairs = []
    for i in range(len(path_list)):
        for j in range(i, len(path_list)):
            pairs.append([path_list[i], path_list[j]])
    if os.path.exists(save_path):
        exitsting = pd.read_csv(save_path, sep=' ', header=None)
        cnt_existing = exitsting[0].value_counts().to_dict()
        cnt_total = Counter(np.array(pairs)[:, 0])
        done = []
        for key in cnt_existing.keys():
            if cnt_existing[key] == cnt_total[key]:
                done.append(key)
        pairs = []
        print("done:", len(done))
        for i in range(len(path_list)):
            if path_list[i] not in done:
                for j in range(i, len(path_list)):
                    pairs.append([path_list[i], path_list[j]])

    pool = Pool(cores)
    pool.map(partial(_set_call_back, pairs=pairs, save_path=save_path), trange(len(pairs)))
    pool.close()
    pool.join()

    info = pd.read_csv(save_path, sep=' ', header=None)
    info.columns = ['p1', 'p2', 'v1', 'v2']
    info = info.drop_duplicates()
    _score = pd.pivot_table(info, index='p1', columns='p2', values='v1')
    _score = _score.fillna(0)
    _score_t = pd.pivot_table(info, index='p2', columns='p1', values='v2')
    _score_t = _score_t.fillna(0)


    score = _score + _score_t

    np.fill_diagonal(score.values, 1)
    score.index.name = None


    score.index = list(score.index)
    score.columns = list(score.columns)
    score = score.loc[score.index.isin(path_list), score.columns.isin(path_list)]
    # score.index = [item.split('/')[-1].split('.')[0] for item in list(score.index)]
    # score.columns = [item.split('/')[-1].split('.')[0] for item in list(score.columns)]

    return score



def main(args):
   

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    template = brain.Template.allen
    t1 = time.time()

    axon_length = projection_batch.compute_projection_parallel(projection_neuron.projection_length,
                                                               args.data_path,
                                                               cores=args.workers,
                                                               template=template,
                                                               annotation=io.ALLEN_ANNOTATION,
                                                               resolution=10,
                                                               save=False)
    axon_length.to_csv(os.path.join(args.save_path, 'axon_length.csv'))

    # topographic_info = projection_batch.compute_projection_parallel(projection_neuron.topographic_projection_info,
    #                                                                 args.data_path,
    #                                                                 cores=args.workers,
    #                                                                 template=template,
    #                                                                 annotation=io.ALLEN_ANNOTATION,
    #                                                                 resolution=10,
    #                                                                 save=False)
    # topographic_info[['neuron', 'x', 'y', 'z', 'region']].to_csv(os.path.join(args.save_path, 'terminal_info.csv'),
    #                                                              index=False)
    print(time.time() - t1)

    # matrix = distance.morphology_matrix(args.data_path, cores=args.workers, save_path=os.path.join(args.save_path, 'scores_record.txt'))
    # # matrix = nblast(args.data_path, cores=args.workers, save_path=os.path.join(args.save_path, 'nblast_scores_record.txt'))
    # matrix.to_csv(os.path.join(args.save_path, 'scores.csv'))

    # cluster_info = cluster.cluster(n_cluster=args.n_clusters,
    #                                feature=cluster.Feature.precomputed,
    #                                matrix=matrix,
    #                                projection=axon_length,
    #                                save=True,
    #                                save_path=args.save_path)



if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="SingleNeuronAnalyseSummary args description")
    parse.add_argument('--data_path', '-d', type=str, default='', help='swc data path')
    parse.add_argument('--save_path', '-p', type=str, default='', help="results save path")
    parse.add_argument('--workers', type=int, default=1,
                       help="The maximum number of processes that can be used to '\
                          'execute the given calls. If None or not given then as many' \
                          worker processes will be created as the machine has processors.")
    parse.add_argument("--n_clusters", '-n', type=int, default=3, help="set cluster number, must be set > 0")

    args = parse.parse_args()
    print(args)
    t1 = time.time()
    main(args)
    print(time.time() - t1)
