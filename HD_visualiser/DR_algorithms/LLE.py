from DR_algorithms.DR_algorithm import DR_algorithm

class LLE(DR_algorithm):
    def __init__(self, algo_name):
        super(LLE, self).__init__(algo_name)
        self.add_int_hyperparameter('n_neighbors', 2, 1000, 1, 5)
        self.add_float_hyperparameter('reg',0.0001, 1., 0.0001, 0.001 )
        self.add_string_hyperparameter('eigen_solver', ["auto","arpack","dense"], "auto")
        self.add_float_hyperparameter('tol (arpack)',   1e-07, 1e-05, 1e-07, 1e-06)
        self.add_int_hyperparameter('max_iter (arpack)', 5, 1000, 1, 100)
        self.add_string_hyperparameter('method', ["standard","hessian","modified","ltsa"], "standard")
        self.add_float_hyperparameter('hessian_tol',   0.00001, 0.001, 0.00001, 0.0001)
        self.add_float_hyperparameter('modified_tol',   1e-13, 1e-11, 1e-13, 1e-12 )
        self.add_string_hyperparameter('neighbors_algorithm', ["auto","brute","kd_tree","ball_tree"], "auto")
        self.add_int_hyperparameter('random_state', 0, 100, 1, None)
        self.add_int_hyperparameter('n_jobs', 1, 16, 1, 4)
        self.embedding = None


    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")
        from sklearn.manifold import LocallyLinearEmbedding
        hparams = self.get_hyperparameters()
        model = LocallyLinearEmbedding(
                                n_components=2,\
                                reg=hparams["reg"],\
                                n_neighbors=hparams["n_neighbors"],\
                                eigen_solver=hparams["eigen_solver"],\
                                tol=hparams["tol (arpack)"],\
                                max_iter=hparams["max_iter (arpack)"],\
                                hessian_tol=hparams["hessian_tol"],\
                                modified_tol=hparams["modified_tol"],\
                                neighbors_algorithm=hparams["neighbors_algorithm"],\
                                random_state=hparams["random_state"],\
                                n_jobs=hparams["n_jobs"],\
                                method=hparams["method"]
                                )
        self.embedding = model.fit_transform(X)

    def transform(self, X, Y):
        return self.embedding
