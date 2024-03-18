from DR_algorithms.DR_algorithm import DR_algorithm

class isomap(DR_algorithm):
    def __init__(self, algo_name):
        super(isomap, self).__init__(algo_name)
        self.add_int_hyperparameter('N_jobs', 1, 16, 1, 1)
        self.add_int_hyperparameter('n_neighbors', 4, 300, 2, 8)
        self.add_int_or_string_hyperparameter('max_iter', 20, 2000, 10, None)
        self.add_string_hyperparameter('path_method', ["auto", "FW", "D"], "auto")
        self.embedding = None

    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")
        hparams = self.get_hyperparameters()
        from sklearn.manifold import Isomap
        model = Isomap(
                        n_neighbors=hparams["n_neighbors"],
                        n_components=2,
                        eigen_solver='auto',
                        tol=0,
                        max_iter=hparams["max_iter"],
                        path_method=hparams["path_method"],
                        neighbors_algorithm='auto',
                        n_jobs=hparams["N_jobs"],
                        metric='minkowski',
                        p=2,
                        metric_params=None
                        )
        self.embedding = model.fit_transform(X)


    def transform(self, X, Y):
        return self.embedding
