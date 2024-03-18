from DR_algorithms.DR_algorithm import DR_algorithm
import numpy as np

class UMAP(DR_algorithm):
    def __init__(self, algo_name):
        super(UMAP, self).__init__(algo_name)
        self.add_int_hyperparameter('n_neighbours', 3, 200, 1, 15)
        self.add_float_hyperparameter('min_dist', 0.001, 0.5, 0.001, 0.1)
        self.add_bool_hyperparameter('supervised', False)
        self.add_string_hyperparameter('metric', ["euclidean", "manhattan", "chebyshev", "canberra", "cosine", "correlation", "hamming", "russellrao", "rogerstanimoto"], "euclidean")
        self.embedding = None

    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")

        print("with this implementations of umap, there can be issues when multiple umaps are launched in parallel. It is advised to wait for the previous umap to finish before launching another")
        try:
            import umap
        except:
            raise Exception("could not import umap. aborting")
        hparams = self.get_hyperparameters()
        if hparams["supervised"]:
            self.embedding = umap.UMAP(\
                                        n_neighbors = hparams["n_neighbours"],\
                                        min_dist = hparams["min_dist"],\
                                        metric = hparams["metric"]\
                                        ).fit_transform(X, y = Y)
        else:
            self.embedding = umap.UMAP(\
                                        n_neighbors = hparams["n_neighbours"],\
                                        min_dist = hparams["min_dist"],\
                                        metric = hparams["metric"]\
                                        ).fit_transform(X)

    def transform(self, X, Y):
        return self.embedding
