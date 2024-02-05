import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from cluster import cluster, Method, Feature, plot_cluster
from projection import projection_neuron, projection_batch
from reader import io, swc, brain
from visualization import *


class Summary:
    def __init__(self, data_path, template=brain.Template.allen, annotation=None, resolution=None, region_path=None,
                 cores=cpu_count() // 2, save_path=os.getcwd()):
        self.data_path = data_path
        self.template = template
        if template == brain.Template.allen:
            self.annotation = io.ALLEN_ANNOTATION
            self.resolution = 10
            self.region_path = io.ALLEN_ROOT_PATH
        else:
            self.annotation = io.read_nrrd(annotation)
            self.resolution = resolution
            self.region_path = region_path
        self.cores = cores
        self.save_path = save_path

    def __check_data(self):
        path_list = swc.read_neuron_path(self.data_path)
        results = []
        with ThreadPoolExecutor(max_workers=self.cores) as executor:
            results = executor.map(swc.check_swc, path_list)
        wrong_swc = len(path_list) - sum(results)
        return len(path_list), sum(results), wrong_swc

    def __get_neuron_info(self, ):
        soma_info = swc.plot_soma_distribution(self.data_path, save=True, save_path=self.save_path)
        return soma_info

    def __get_axon_length(self):
        self.axon_length = projection_batch.compute_projection_parallel(projection_neuron.projection_length,
                                                                        self.data_path,
                                                                        cores=self.cores,
                                                                        template=self.template,
                                                                        annotation=self.annotation,
                                                                        resolution=self.resolution,
                                                                        save=False)
        return self.axon_length

    def __get_cluster_info(self, n_clusters):
        self.__get_axon_length()
        self.axon_length.to_csv(os.path.join(self.save_path, 'axon_length.csv'))
        cluster_info = cluster(n_clusters,
                               template=self.template,
                               method=Method.hierarchy,
                               feature=Feature.morphology,
                               projection=self.axon_length,
                               data_path=self.data_path,
                               save=True,
                               save_path=self.save_path)
        plot_cluster(cluster_info, show=False, save_path=self.save_path, region_path=self.region_path, region_opacity=0.2, bgcolor='white')
        return cluster_info

    def __get_topographic_info(self):
        topographic_info = projection_batch.compute_projection_parallel(projection_neuron.topographic_projection_info,
                                                                        self.data_path,
                                                                        cores=self.cores,
                                                                        template=self.template,
                                                                        annotation=self.annotation,
                                                                        resolution=self.resolution,
                                                                        save=False)
        topographic_info.to_csv('topo.csv', index=0)
        show_data = projection_vis.plot_topographic_projection(topographic_info,
                                                               self.template,
                                                               threshold=2,
                                                               p_threshold=0.05,
                                                               save=True,
                                                               save_path=self.save_path)
        return show_data

    def summary_pipeline(self, n_clusters):
        # neuron info summary
        neuron_num, is_valid, not_valid = self.__check_data()
        print('neuron num:%2d, valid neuron num:%2d, fix %2d neurons' % (neuron_num, is_valid, not_valid))
        #
        self.__get_neuron_info()
        # cluster info
        if n_clusters <= 0:
            print('illegal cluster num settings, the n_cluster must be set > 0')
            return
        if n_clusters > neuron_num:
            print('illegal cluster num settings, the n_cluster is %d, the number of neuron is %d' % (
                n_clusters, neuron_num))
            return
        cluster_info = self.__get_cluster_info(n_clusters)
        topographoc_info = self.__get_topographic_info()
        return


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="args init")
    parse.add_argument('--data_path', '-d', type=str, default='', help='swc data path')
    parse.add_argument('--save_path', '-p', type=str, default='', help="results save path")
    parse.add_argument("--template", '-t', type=int, default=0,
                       help="brain altas, 0-allen mouse brain altas(2017) or 1-customized, defalut: 0")
    parse.add_argument('--annotation', '-a', type=str, default='',
                       help="when set template as 1, set annotation path (.nrrd)")
    parse.add_argument("--resolution", '-r', type=int, default=10, help="the altas annotation resolution")
    parse.add_argument('--workers', type=int, default=1,
                       help="The maximum number of processes that can be used to '\
                       'execute the given calls. If None or not given then as many' \
                       worker processes will be created as the machine has processors.")
    parse.add_argument("--n_clusters", '-n', type=int, default=3, help="set cluster number, must be set > 0")
    args = parse.parse_args()
    if args.template == 0:
        template = brain.Template.allen
    else:
        template = brain.Template.customized
        if not args.annotation:
            print('annotation must be set since template is customized.')
    summary = Summary(data_path=args.data_path,
                      template=template,
                      annotation=args.annotation,
                      resolution=args.resolution,
                      cores=args.workers,
                      save_path=args.save_path)
    summary.summary_pipeline(args.n_clusters)
