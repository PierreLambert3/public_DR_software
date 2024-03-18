from DR_algorithms.DR_algorithm import DR_algorithm

class parametric_ms_tSNE(DR_algorithm):
    def __init__(self, algo_name):
        super(parametric_ms_tSNE, self).__init__(algo_name)
        self.add_string_hyperparameter('svd_solver', ["auto", "full", "arpack", "randomized"], "auto")
        self.add_bool_hyperparameter('whiten', True)
        self.add_int_hyperparameter('random_state', 0, 100, 1, None)
        self.add_int_or_string_hyperparameter('iterated_power', 0, 200, 3,  "auto")
        self.add_float_hyperparameter('tol', 0., 1., 0.05, 0)
        self.embedding = None

    def fit(self, progress_listener, X, Y):
        hparams = self.get_hyperparameters()


    def transform(self, X, Y):
        return self.embedding
