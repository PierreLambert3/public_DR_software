
import numpy as np


def get_DR_algorithm(algo_name):
    if algo_name == 'PCA':
        from DR_algorithms.PCA import PCA
        return PCA(algo_name)
    elif algo_name == 'PCA (iterative)':
        from DR_algorithms.PCA_iterative import PCA_iterative
        return PCA_iterative(algo_name)
    elif algo_name == 'MDS':
        from DR_algorithms.MDS import MDS
        return MDS(algo_name)
    elif algo_name == 'SQuaD-MDS':
        from DR_algorithms.SQuaD_MDS import SQuaD_MDS
        return SQuaD_MDS(algo_name)
    elif algo_name == 'SQuaD-MDS + tSNE':
        from DR_algorithms.SQuaD_MDS_tSNE import SQuaD_MDS_tSNE
        return SQuaD_MDS_tSNE(algo_name)
    elif algo_name == 'tSNE':
        from DR_algorithms.tSNE import tSNE
        return tSNE(algo_name)
    elif algo_name == 'multiscale-tSNE':
        from DR_algorithms.multiscale_NE import multiscale_NE
        return multiscale_NE(algo_name)
    elif algo_name == 'myTest':
        from DR_algorithms.myTest import myTest
        return myTest(algo_name)
    elif algo_name == 'Laplacian eigenmaps':
        from DR_algorithms.Laplacian_eigenmaps import Laplacian_eigenmaps
        return Laplacian_eigenmaps(algo_name)
    elif algo_name == 'LLE':
        from DR_algorithms.LLE import LLE
        return LLE(algo_name)
    elif algo_name == 'isomap':
        from DR_algorithms.isomap import isomap
        return isomap(algo_name)
    elif algo_name == 'autoencoder':
        from DR_algorithms.autoencoder import autoencoder
        return autoencoder(algo_name)
    elif algo_name == 'DL_method':
        from DR_algorithms.DL_method import DL_method
        return DL_method(algo_name)
    elif algo_name == 'var-autoencoder':
        from DR_algorithms.var_autoencoder import var_autoencoder
        return var_autoencoder(algo_name)
    elif algo_name == 'UMAP':
        from DR_algorithms.UMAP import UMAP
        return UMAP(algo_name)
    elif algo_name == 'FItSNE':
        from DR_algorithms.FItSNE import FItSNE
        return FItSNE(algo_name)
    elif algo_name == 'multiscale-parametric tSNE':
        from DR_algorithms.parametric_ms_tSNE import parametric_ms_tSNE
        return parametric_ms_tSNE(algo_name)
    elif algo_name == 'cat-SNE':
        from DR_algorithms.catSNE import catSNE
        return catSNE(algo_name)
    else:
        print("an unknown algorithm was requested, please add its import and return an instance of the algorithm in the big if/else of the function get_DR_algorithm(algo_name) in file DR_algorithms.py")
        1/0

class DR_algorithm():
    def __init__(self, algo_name):
        self.name = algo_name
        self.dataset_name = None
        self.proj_name    = None
        self.fitted = False
        self.deleted = False
        self.hyperparameters = Hyperparameters()

    def compute_stress(self, Xhd, method_name):
        pass
        # from scipy.spatial.distance import pdist
        # Xld = self.embedding
        # N, M = Xhd.shape
        # N_sample = min(N-1, 3000)
        # perms = np.arange(Xld.shape[0])
        # sample = np.random.choice(perms, size=N_sample, replace=False)
        #
        # Xhd = Xhd / np.std(Xhd)
        # Xld = Xld / np.std(Xld)
        #
        # Xhd_D = pdist(Xhd[sample])
        # Xld_D = pdist(Xld[sample])
        #
        # Nb_dist = Xhd_D.shape[0]
        # sum_residuals_abs    = 0.
        # sum_residuals_square = 0.
        # for i in range(Nb_dist):
        #     residual = Xhd_D[i] - Xld_D[i]
        #     sum_residuals_abs    += np.abs(residual)
        #     sum_residuals_square += residual*residual
        # stress_abs = np.round(sum_residuals_abs / Nb_dist, 3)
        # stress_sqr = np.round(np.sqrt(sum_residuals_square / Nb_dist), 3)
        # print(method_name, "stress (abs) : ", stress_abs,'    (squared) : ', stress_sqr)

    def tag(self, dataset_name, proj_name):
        self.dataset_name = dataset_name
        self.proj_name    = proj_name

    def fit(self, progress_listener, X, Y, is_dists):
        hparams = self.get_hyperparameters()
        self.fitted = True

    def transform(self, X, Y):
        if X.shape[1] <= 2:
            print("X is alread of dimension smaller or equal to 2")
            return X
        else:
            return X[:, :2]

    def fit_transform(self, progress_listener, X, Y, is_dists):
        self.fit(progress_listener, X, Y, is_dists)
        return self.transform(X, Y)

    def set_hparams(self, hparams_dict):
        self.hyperparameters.values = hparams_dict

    def get_hyperparameters(self):
        return self.hyperparameters.values

    def get_hyperparameter_schematics(self):
        return self.hyperparameters.schematics

    def get_hyperparameter_schematics_copy(self):
        scheme = self.hyperparameters.schematics
        copy_scheme = {}
        for key in scheme:
            copy_scheme[key] = [ e for e in scheme[key]]
        return copy_scheme

    def add_float_hyperparameter(self, name, min_value, max_value, step, default_value):
        self.hyperparameters.add_float_hyperparameter(name, min_value, max_value, step, default_value)

    def add_int_hyperparameter(self, name, min_value, max_value, step, default_value):
        self.hyperparameters.add_int_hyperparameter(name, min_value, max_value, step, default_value)

    def add_int_or_string_hyperparameter(self, name, min_value, max_value, step, default_value): # default can be a string
        self.hyperparameters.add_int_or_string_hyperparameter(name, min_value, max_value, step, default_value)

    def add_string_hyperparameter(self, name, possible_values, default_value):
        self.hyperparameters.add_string_hyperparameter(name, possible_values, default_value)

    def add_bool_hyperparameter(self, name, default_value):
        self.hyperparameters.add_bool_hyperparameter(name, default_value)


class Hyperparameters():
    def __init__(self):
        self.values     = {}
        self.schematics = {}

    def add_float_hyperparameter(self, name, min_value, max_value, step, default_value):
        self.schematics[name] = ["float",  min_value, max_value, step, default_value]
        self.values[name]     = default_value

    def add_int_hyperparameter(self, name, min_value, max_value, step, default_value):
        self.schematics[name] = ["int",  min_value, max_value, step, default_value]
        self.values[name]     = default_value

    def add_int_or_string_hyperparameter(self, name, min_value, max_value, step, default_value): # default can be a string
        self.schematics[name] = ["int-str",  min_value, max_value, step, default_value]
        self.values[name]     = default_value

    def add_string_hyperparameter(self, name, possible_values, default_value):
        self.schematics[name] = ["str",  name, possible_values, default_value]
        self.values[name]     = default_value

    def add_bool_hyperparameter(self, name, default_value):
        self.schematics[name] = ["bool",  name, default_value]
        self.values[name]     = default_value
