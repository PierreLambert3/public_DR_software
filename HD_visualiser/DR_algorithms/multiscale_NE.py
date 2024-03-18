from DR_algorithms.DR_algorithm import DR_algorithm
import numpy as np

'''
as implemented in https://github.com/cdebodt/Fast_Multi-scale_NE
from the paper: "Fast Multiscale Neighbor Embedding," in IEEE Transactions on Neural Networks and Learning Systems, 2020, doi: 10.1109/TNNLS.2020.3042807
by: C. de Bodt, D. Mulders, M. Verleysen and J. A. Lee,

the non-accelerated version is from the paper: Multi-scale similarities in stochastic neighbour embedding: Reducing dimensionality while preserving both local and global structure
by: John A. Lee and Diego H. Peluffo-Ordnez and Michel Verleysen

'''

class multiscale_NE(DR_algorithm):
    def __init__(self, algo_name):
        import sys, os
        sys.path.insert(0, os.path.abspath('Cpython_implementations'))
        import cython_implem

        super(multiscale_NE, self).__init__(algo_name)
        self.add_string_hyperparameter('init', ["random uniform", "PCA"], "PCA")
        self.add_int_hyperparameter('random_state', 0, 100, 1, 42)
        self.add_bool_hyperparameter('t_distributed', True)
        self.add_bool_hyperparameter('fast method', True)
        self.add_float_hyperparameter('angle (Barnes-Hut)', 0.05, 0.95, 0.0125, 0.75)
        self.add_bool_hyperparameter('(if fast and Student) biaised sampling', True)
        self.add_int_hyperparameter('(if fast) nb of resamplings',  1, 6, 1, 1)
        self.add_int_hyperparameter('optim. niter_max',   5, 1000, 5, 30)
        self.add_float_hyperparameter('optim. gtol', 0.1e-5, 10e-5, 0.1e-5, 1e-5)
        self.add_float_hyperparameter('optim. ftol', 2.2204460492503131e-10, 2.2204460492503131e-08, 2.2204460492503131e-10, 2.2204460492503131e-09)
        self.add_int_hyperparameter('optim. max_linesearch_steps',   5, 130, 1, 50)
        self.add_int_hyperparameter('optim. max_var_correction',  3, 90, 1, 10)
        self.embedding = None


    def fit(self, progress_listener, X, Y):
        hparams = self.get_hyperparameters()
        try:
            import sys, os
            sys.path.insert(0, os.path.abspath('Cpython_implementations'))
            import cython_implem
        except:
            raise Exception("could not import cython_implem for multiscale_NE. Make sure the files in Cpython_implementations are compiled and that the \'Cpython_implementations\' directory is in the same directory as main.py. This Exception was raised from fit() in multiscale_NE.py")

        N, M = X.shape
        X_HD_1D = np.ascontiguousarray(a=np.reshape(a=X, newshape=N*M, order='C'), dtype=np.float64)
        X_LD_1D = np.ascontiguousarray(a=np.reshape(a=init_LD(X, hparams['init'], hparams['random_state']), newshape=N*2, order='C'), dtype=np.float64)
        if not hparams['fast method']:
            if hparams['t_distributed']:
                cython_implem.mstsne_implem(X_HD_1D, X_LD_1D, N, M, 2, hparams['optim. niter_max'], hparams['optim. gtol'], hparams['optim. ftol'], hparams['optim. max_linesearch_steps'], hparams['optim. max_var_correction'], 1)
            else:
                cython_implem.mssne_implem(X_HD_1D, X_LD_1D, N, M, 2, True, hparams['optim. niter_max'], hparams['optim. gtol'], hparams['optim. ftol'], hparams['optim. max_linesearch_steps'], hparams['optim. max_var_correction'], 1)
        else:
            if hparams['t_distributed']:
                if hparams['(if fast and Student) biaised sampling']:
                    from sklearn.neighbors import NearestNeighbors
                    neigh_small       = NearestNeighbors(n_neighbors=int(0.1*N), radius=0.4)
                    neigh_small.fit(X)
                    neighbours_small  = neigh_small.kneighbors(X, return_distance=False)[:, 1:].astype(np.int32)
                    trace_coefs_small = np.ones(neighbours_small.shape)

                    cython_implem.fmstsne_implem_with_trace(1, trace_coefs_small, np.ones((N,)), neighbours_small, np.ones((N), dtype = np.int32)*N, X_HD_1D, X_LD_1D, N, M, 2, True, int(hparams['(if fast) nb of resamplings']), float(hparams['angle (Barnes-Hut)']), int(hparams['optim. niter_max']), float(hparams['optim. gtol']), float(hparams['optim. ftol']), int(hparams['optim. max_linesearch_steps']), int(hparams['optim. max_var_correction']), 1, int(hparams['random_state']))
                else:
                    cython_implem.fmstsne_implem(X_HD_1D, X_LD_1D, N, M, 2, True, int(hparams['(if fast) nb of resamplings']), float(hparams['angle (Barnes-Hut)']), int(hparams['optim. niter_max']), float(hparams['optim. gtol']), float(hparams['optim. ftol']), int(hparams['optim. max_linesearch_steps']), int(hparams['optim. max_var_correction']), 1, int(hparams['random_state']))
            else:
                cython_implem.fmssne_implem(X_HD_1D, X_LD_1D, N, M, 2, True, 1, 1, float(hparams['angle (Barnes-Hut)']), int(hparams['optim. niter_max']), float(hparams['optim. gtol']), float(hparams['optim. ftol']), int(hparams['optim. max_linesearch_steps']), int(hparams['optim. max_var_correction']), 1, int(hparams['random_state']), False)

        self.embedding = np.reshape(a=X_LD_1D, newshape=(N, 2), order='C')

    def transform(self, X, Y):
        return self.embedding

def init_LD(X, init_type, seed=None):
    if init_type == "PCA":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, whiten=True, copy=True).fit_transform(X)
    else:
        return np.random.uniform(size=(X.shape[0], 2))
