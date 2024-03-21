import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from .cluster import cluster, Method, Feature, plot_cluster
from .projection import projection_batch, projection_neuron
from .reader import brain, io, swc
from .visualization import projection_vis
from .web_summary import build_web_summary


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
        if not os.path.exists(save_path):
            os.mkdir(save_path)
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
        show_data = projection_vis.plot_topographic_projection(topographic_info,
                                                               self.template,
                                                               threshold=10,
                                                               p_threshold=0.05,
                                                               save=True,
                                                               save_path=self.save_path)
        return show_data

    def summary_pipeline(self, n_clusters):
        # neuron info summary
        neuron_num, is_valid, not_valid = self.__check_data()
        print('neuron num:%2d, valid neuron num:%2d, fix %2d neurons' % (neuron_num, is_valid, not_valid))
        #
        soma_info = self.__get_neuron_info()
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
        build_web_summary([neuron_num, is_valid, not_valid],
                          soma_info, cluster_info,
                          topographoc_info, self.template, self.save_path)
        print("web summary path: %s\n" \
              "cluster result path: %s\n"\
              "axon length path: %s\n"\
              "soma distribution path: %s\n"\
              "projection pattern path: %s\n"\
              "tsne path: %s\n"\
              "topographic projection path:%s\n" % (os.path.join(self.save_path, 'Single-Neuron-Report.html'),
                                                    os.path.join(self.save_path, 'cluster_results.csv'),
                                                    os.path.join(self.save_path, 'axon_length.csv'),
                                                    os.path.join(self.save_path, 'soma_distribution.png'),
                                                    os.path.join(self.save_path, 'projection_pattern.png'),
                                                    os.path.join(self.save_path, 'tsne.png'),
                                                    os.path.join(self.save_path, "topographic_projection.png")))
        return


