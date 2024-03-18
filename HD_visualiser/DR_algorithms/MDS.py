from DR_algorithms.DR_algorithm import DR_algorithm

class MDS(DR_algorithm):
    def __init__(self, algo_name):
        super(MDS, self).__init__(algo_name)
        self.add_bool_hyperparameter('metric MDS', True)
        self.add_bool_hyperparameter('PCA init', True)
        self.add_int_hyperparameter('n_init (ignored if PCA init)', 1, 30, 1, 4)
        self.add_int_hyperparameter('max_iter', 40, 1000, 10, 300)
        self.add_float_hyperparameter('eps', 0.1e-3, 10e-3, 0.1e-3, 1e-3)
        self.add_int_hyperparameter('n_jobs', 1, 16, 1, 4)
        self.add_bool_hyperparameter('use squared euclidean', False)
        self.add_int_or_string_hyperparameter('random_state', 1, 100, 1, None)
        self.embedding = None

    def PCAinit_MDS(self, X, Y, hparams):
        from sklearn.metrics.pairwise import pairwise_distances
        from sklearn.decomposition    import PCA
        from sklearn.manifold         import smacof
        Xpc = PCA(n_components = 2).fit_transform(X)
        if hparams['use squared euclidean']:
            self.embedding = smacof(pairwise_distances(X)**2, n_components=2, init=Xpc, n_init=1)[0]
        else:
            self.embedding = smacof(pairwise_distances(X), n_components=2, init=Xpc, n_init=1)[0]

    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")
        hparams = self.get_hyperparameters()
        if hparams['PCA init']:
            self.PCAinit_MDS(X, Y, hparams)
        else:
            from sklearn.manifold import MDS as sklearn_MDS
            if hparams['use squared euclidean']:
                model = sklearn_MDS(
                            n_components=2,\
                            metric=hparams['metric MDS'],\
                            n_init=hparams["n_init (ignored if PCA init)"],\
                            max_iter=hparams["max_iter"],\
                            verbose=0,\
                            n_jobs=hparams["n_jobs"],\
                            random_state=hparams["random_state"],\
                            dissimilarity="euclidean",\
                            eps=hparams["eps"]
                            )
                model.fit(X)
            else:
                from sklearn.metrics.pairwise import pairwise_distances
                model = sklearn_MDS(
                            n_components=2,\
                            metric=hparams['metric MDS'],\
                            n_init=hparams["n_init (ignored if PCA init)"],\
                            max_iter=hparams["max_iter"],\
                            verbose=0,\
                            n_jobs=hparams["n_jobs"],\
                            random_state=hparams["random_state"],\
                            dissimilarity="precomputed",\
                            eps=hparams["eps"]
                            )
                model.fit(pairwise_distances(X)**2)
            self.embedding = model.embedding_
        super(MDS, self).compute_stress( X, " MDS ")

    def transform(self, X, Y):
        return self.embedding
