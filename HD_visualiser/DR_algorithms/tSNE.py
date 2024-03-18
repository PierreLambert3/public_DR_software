from DR_algorithms.DR_algorithm import DR_algorithm
import numpy as np
from scipy.linalg import norm
import scipy
from sklearn.manifold import _barnes_hut_tsne, TSNE

class tSNE(DR_algorithm):
    def __init__(self, algo_name):
        super(tSNE, self).__init__(algo_name)
        self.add_string_hyperparameter('metric',["euclidean", "cityblock", "cosine"], "euclidean")
        self.add_bool_hyperparameter('test variant', False)
        self.add_float_hyperparameter('perplexity', 1., 600., 1., 30.)
        self.add_float_hyperparameter('early_exaggeration', 1., 50., 0.5, 12.)
        self.add_float_hyperparameter('learning_rate', 1., 1200., 1., 200.)
        self.add_int_hyperparameter('n_iter', 100, 5000, 25, 1000)
        self.add_float_hyperparameter('min_grad_norm', 0.1e-7, 10e-7, 0.1e-7, 1e-7)
        self.add_string_hyperparameter('init',["pca", "random"], "pca")
        self.add_int_hyperparameter('random_state', 1, 100, 1, None)
        self.add_string_hyperparameter('method',["barnes_hut", "exact"], "barnes_hut")
        self.add_float_hyperparameter('angle (Barnes-Hut)', 0.05, 0.95, 0.025, 0.5)
        self.add_int_hyperparameter('n_jobs', 1, 16, 1, 4)
        self.embedding = None



    def fit(self, progress_listener, X, Y, is_dists):
        hparams = self.get_hyperparameters()
        N, M = X.shape
        if is_dists:
            from sklearn.manifold import TSNE
            # print("model need to be adapted for using distances as input matrix")
            # raise Exception("dist matrix ")
            print("X SHAPE : \n\n\n ", X.shape)
            flat_X_LD = TSNE(n_components=2, perplexity = hparams["perplexity"], metric='precomputed', init='random').fit_transform(X)
        else:
            show_progress = (None not in [progress_listener, self.dataset_name, self.proj_name])
            if not show_progress:
                flat_X_LD = TSNE(n_components=2, perplexity = hparams["perplexity"], init=hparams['init']).fit_transform(X)
            else:
                P = joint_P(X, hparams["perplexity"], hparams["method"], hparams["n_jobs"], hparams['test variant'])
                if hparams['init'] == "pca":
                    from sklearn.decomposition import PCA
                    X_LD = PCA(n_components=2, whiten=True, copy=True).fit_transform(X)*1e-3
                else:
                    X_LD = (np.random.uniform(size=(N*2))*1e-3).reshape((N, 2))

                if hparams["method"] == "exact":
                    obj_func = KL_divergeance
                else:
                    obj_func = KL_divergeance_BH

                hparams["P"] = P
                hparams["degrees_of_freedom"] = 1
                hparams["n_samples"] = N
                n_iter = hparams["n_iter"]
                nb_iter_exploration = int(n_iter*0.25)
                exploration_momentum = 0.6
                end_momentum = 0.85

                # import sys
                # sys.exit()


                P *= hparams["early_exaggeration"]
                flat_X_LD = X_LD.ravel()
                flat_X_LD, _, it = self.gradient_descent(obj_func, flat_X_LD, hparams, progress_listener,
                                    iter=0, n_iter=nb_iter_exploration,
                                    n_iter_check=3, n_iter_without_progress=300,
                                    momentum=exploration_momentum, learning_rate=hparams["learning_rate"], min_gain=0.01,
                                    min_grad_norm=hparams["min_grad_norm"], verbose=0, show_convergence=show_progress)

                remaining_iter = n_iter - nb_iter_exploration
                P /= hparams["early_exaggeration"]
                flat_X_LD, _, it = self.gradient_descent(obj_func, flat_X_LD, hparams, progress_listener,
                                    iter=it+1, n_iter=remaining_iter,
                                    n_iter_check=3, n_iter_without_progress=300,
                                    momentum=end_momentum, learning_rate=hparams["learning_rate"], min_gain=0.01,
                                    min_grad_norm=hparams["min_grad_norm"], verbose=0, show_convergence=show_progress)

        self.embedding = flat_X_LD.reshape((N, 2))
        super(tSNE, self).compute_stress(X, " tSNE ")

    def transform(self, X, Y):
        return self.embedding


    def gradient_descent(self, objective, flat_X_LD, args, listener, iter, n_iter,
                          n_iter_check=1, n_iter_without_progress=300,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01,
                          min_grad_norm=1e-7, verbose=0 , show_convergence=False):
        P = args["P"]
        degrees_of_freedom = args["degrees_of_freedom"]
        n_samples = args["n_samples"]

        params = flat_X_LD.copy().ravel()
        update = np.zeros_like(params)
        gains  = np.ones_like(params)
        error  = np.finfo(float).max
        best_error = np.finfo(float).max
        best_iter = i = iter

        import scipy
        for i in range(iter, n_iter):
            check_convergence = (i + 1) % n_iter_check == 0

            error, grad = objective(params, P, degrees_of_freedom, n_samples, 2,
                                  0, check_convergence)



            grad_norm = norm(grad)

            inc = update * grad < 0.0 # indexes where same direction
            dec = np.invert(inc)
            gains[inc] += 0.2 # same direction: speed up
            gains[dec] *= 0.8 # directional change: smaller steps
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            params += update
            if check_convergence:
                if error < best_error:
                    best_error = error
                    best_iter = i
                elif i - best_iter > n_iter_without_progress:
                    break
                if grad_norm <= min_grad_norm:
                    break
                if show_convergence:
                    listener.notify((self.dataset_name, self.proj_name, params.reshape((n_samples, 2)), self), [])

        return params, error, i


MACHINE_EPSILON = np.finfo(np.double).eps

def joint_probabilities(D, target_PP):
    from sklearn.manifold._utils import _binary_search_perplexity
    conditional_P = _binary_search_perplexity(D, target_PP, 0)
    P = conditional_P + conditional_P.T
    # return P/(D.shape[0]*2)
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P

def joint_probabilities_nn(D, target_PP):
    from sklearn.manifold._utils import _binary_search_perplexity
    from scipy.sparse import csr_matrix
    D.sort_indices()
    N = D.shape[0]
    D_data = D.data.reshape(N, -1)
    D_data = D_data.astype(np.float32, copy=False)
    conditional_P = _binary_search_perplexity(D_data, target_PP, 0)
    assert np.all(np.isfinite(conditional_P))

    # Symmetrize the joint probability distribution using sparse operations
    # ^ car on est parti depuis du knn: pas symmetrique!
    P = csr_matrix((conditional_P.ravel(), D.indices, D.indptr), shape=(N, N))
    P = P + P.T
    # return P/(D.shape[0]*2)

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    return P

# def joint_P(X, PP, method, N_jobs):
#     from sklearn.neighbors import NearestNeighbors
#     N, dim = X.shape
#     if method == "exact":
#         D = pairwise_distances(X, metric="euclidean", squared=True).astype(np.float32, copy=False)
#         P = joint_probabilities(D, PP)
#     else:
#         n_neighbors = min(N - 1, int(3.*PP + 1))
#         knn = NearestNeighbors(algorithm='auto', n_jobs=N_jobs, n_neighbors=n_neighbors, metric="euclidean")
#         knn.fit(X)
#         D_nn = knn.kneighbors_graph(mode='distance')
#
#         # D_nn[D_nn > 0] -= np.min(D_nn[D_nn > 0])*0.9
#
#         D_nn.data **= 2
#         del knn
#         P = joint_probabilities_nn(D_nn, PP)
#     return P

import numba
@numba.jit(nopython=True, fastmath=True)
def asjadkdsa(N, X, neighs, intensities):
    # X_int = X * intensities
    X_int = X * intensities
    tmp = np.zeros((X.shape[0], neighs.shape[1]-1))
    for obs in range(N):
        if obs % 20 == 0:
            print(obs)
        Xi = X_int[obs]
        Xi_neighs = neighs[obs][1:]
        # tmp[obs] += np.sum((Xi - X_int[Xi_neighs])**2, axis=1)
        su = np.sum((Xi - X[Xi_neighs]*(intensities[Xi_neighs]*0.5 + intensities[obs]*0.5))**2, axis=1)
        tmp[obs] += su
    return np.sqrt(tmp)

def dist_local(X, knn, n_neighbors):
    D_nn = knn.kneighbors_graph(mode='distance')
    N, M = X.shape[0], X.shape[1]
    stds = np.std(X, axis = 0)
    neighs = knn.kneighbors(X=X, n_neighbors=n_neighbors+1, return_distance=False)
    gaussian = np.exp(-(np.arange(n_neighbors+1)**2)/(2*((0.5*(n_neighbors+1))**2)))
    sum_gauss = np.sum(gaussian)
    intensities = np.zeros_like(X) + 1e-4
    for obs in range(N):
        sample_idxs = neighs[obs]
        sample = X[sample_idxs]
        means = np.mean(sample, axis=0) # mean of each feature on the sample
        dev = np.clip((sample - means)**2, a_min = 1e-8, a_max = None)
        dev = dev * gaussian[:, None]
        sample_stds = np.sqrt(np.sum(dev, axis=0)/sum_gauss)
        sample_stds -= np.min(sample_stds)
        int = 1 - np.exp(-(sample_stds**2) / (2 * np.std(sample_stds)**2))
        int /= np.sum(int)
        intensities[obs] += int

    intensities /= np.sum(intensities, axis=1)[:, None]

    tmp = asjadkdsa(N, X, neighs, intensities)
    for obs in range(N):
        # D_nn[obs, neighs[obs][1:]] += tmp[obs]
        # D_nn[obs, neighs[obs][1:]] = 0.1*D_nn[obs, neighs[obs][1:]] + 0.9*tmp[obs]
        D_nn[obs, neighs[obs][1:]] = 0.001*D_nn[obs, neighs[obs][1:]] + 0.999*tmp[obs]

    # for obs in range(N):
    #     for nei in neighs[obs]:
    #         v = (D_nn[obs, nei] + D_nn[nei, obs]) / 2
    #         D_nn[obs, nei] = v
    #         D_nn[nei, obs] = v
    return D_nn



# t-SNE's joint P matrix
def joint_P(X, PP, method, N_jobs, do_variant):
    from sklearn.neighbors import NearestNeighbors
    N, dim = X.shape
    n_neighbors = min(N - 1, int(3.*PP + 1))
    knn = NearestNeighbors(algorithm='auto', n_jobs=N_jobs, n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X)

    if do_variant:
        D_nn = dist_local(X, knn, n_neighbors)
    else:
        D_nn = knn.kneighbors_graph(mode='distance')


    D_nn.data **= 2
    del knn
    P = joint_probabilities_nn(D_nn, PP)
    return P

def KL_divergeance(flat_X_LD, P, degrees_of_freedom, n_samples, n_components,\
                    skip_num_points, compute_error):
    # Q is a heavy-tailed distribution: Student's t-distribution
    X_embedded = flat_X_LD.reshape(n_samples, n_components)
    dist = pdist(X_embedded, "sqeuclidean")
    if degrees_of_freedom > 1:
        dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    if compute_error:
        kl_divergence = 2.0 * np.dot(
            P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    grad = np.zeros((n_samples, n_components), dtype=flat_X_LD.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad

def KL_divergeance_BH(flat_X_LD, P, degrees_of_freedom, n_samples, n_components,
                      skip_num_points, compute_error,
                      angle=0.5, verbose=False,  num_threads=1):
    flat_X_LD  = flat_X_LD.astype(np.float32, copy=False)
    X_embedded = flat_X_LD.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr,
                                      grad, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error,
                                      num_threads=num_threads)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad
