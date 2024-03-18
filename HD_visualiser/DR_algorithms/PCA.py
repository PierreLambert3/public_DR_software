from DR_algorithms.DR_algorithm import DR_algorithm

class PCA(DR_algorithm):
    def __init__(self, algo_name):
        super(PCA, self).__init__(algo_name)
        self.add_string_hyperparameter('svd_solver', ["auto", "full", "arpack", "randomized"], "auto")
        self.add_bool_hyperparameter('whiten', True)
        self.add_int_hyperparameter('random_state', 0, 100, 1, None)
        self.add_int_or_string_hyperparameter('iterated_power', 0, 200, 3,  "auto")
        self.add_float_hyperparameter('tol', 0., 1., 0.05, 0)
        self.model = None

    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")
        hparams = self.get_hyperparameters()
        import sklearn.decomposition
        self.model = sklearn.decomposition.PCA(n_components=2,\
                                whiten=hparams["whiten"],\
                                svd_solver=hparams["svd_solver"],\
                                tol=hparams["tol"],\
                                iterated_power=hparams["iterated_power"],\
                                random_state=hparams["random_state"], copy=True)
        self.model.fit(X)


    def transform(self, X, Y):
        Xld = self.model.transform(X)
        self.embedding = Xld
        super(PCA, self).compute_stress( X , ' PCA ')
        return Xld
