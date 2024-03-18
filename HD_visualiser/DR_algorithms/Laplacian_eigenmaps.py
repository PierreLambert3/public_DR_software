from DR_algorithms.DR_algorithm import DR_algorithm

class Laplacian_eigenmaps(DR_algorithm):
    def __init__(self, algo_name):
        super(Laplacian_eigenmaps, self).__init__(algo_name)
        self.add_string_hyperparameter('affinity', ["nearest_neighbors","rbf"], "nearest_neighbors")
        self.add_bool_hyperparameter('automatic gamma', True)
        self.add_float_hyperparameter('gamma (if not automatic)', 1e-4, 1e-1, 1e-4, 1e-3)
        self.add_int_or_string_hyperparameter('n_neighbors', 2, 1000, 1, None)
        self.add_string_hyperparameter('eigen_solver', ["arpack", "lobpcg", "amg"], "arpack")
        self.add_int_hyperparameter('random_state', 0, 100, 1, None)
        self.add_int_hyperparameter('n_jobs', 1, 16, 1, 4)
        self.embedding = None

    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")
        from sklearn.manifold import SpectralEmbedding
        hparams = self.get_hyperparameters()
        gamma = None
        if not hparams['automatic gamma']:
            gamma = hparams['gamma (if not automatic)']

        model = SpectralEmbedding(
                                n_components=2,\
                                affinity=hparams["affinity"],\
                                gamma=gamma,\
                                eigen_solver=hparams["eigen_solver"],\
                                n_neighbors=hparams["n_neighbors"],\
                                n_jobs=hparams["n_jobs"],\
                                random_state=hparams["random_state"]
                                )
        self.embedding = model.fit_transform(X)

    def transform(self, X, Y):
        return self.embedding
