from DR_algorithms.DR_algorithm import DR_algorithm
import numpy as np
import numba
from scipy.spatial.distance import squareform, pdist

'''
as implemented in https://github.com/cdebodt/cat-SNE/blob/master/catsne.py
from : de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414)
'''

class catSNE(DR_algorithm):
    def __init__(self, algo_name):
        super(catSNE, self).__init__(algo_name)
        self.add_bool_hyperparameter('show progress', True)
        self.add_float_hyperparameter('theta', 0.5, .99, 0.01, 0.85)
        self.add_int_hyperparameter('n iter', 100, 10000, 100, 1000)
        self.add_string_hyperparameter('init', ["ran", "pca"], "ran")
        self.embedding = None

    def fit(self, progress_listener, X, Y):
        hparams = self.get_hyperparameters()
        show_progress         = (None not in [progress_listener, self.dataset_name, self.proj_name]) and hparams["show progress"]
        self.embedding, _ = self.catsne(progress_listener, show_progress, X_hds=X, labels=Y, theta=hparams['theta'], init=hparams['init'], dim_lds=2, nit_max=hparams['n iter'], rand_state=np.random.RandomState(0), hd_metric='euclidean')

    def transform(self, X, Y):
        return self.embedding



    def catsne(self, progress_listener, show_progress, X_hds, labels, theta=0.9, init='ran', dim_lds=2, nit_max=1000, rand_state=None, hd_metric='euclidean', D_hd_metric=None, gtol=10.0**(-5.0), ftol=10.0**(-8.0), eef=4, eei=100, mom_init=0.5, mom_fin=0.8, mom_t=250):
        """
        Apply cat-SNE to reduce the dimensionality of a data set by accounting for class labels.
        Euclidean distance is employed in the LDS, as in t-SNE.
        In:
        - X_hds: 2-D numpy array of floats with shape (N, M), containing the HD data set, with one row per example and one column per dimension. N is hence the number of data points and M the dimension of the HDS. It is assumed that the rows of X_hds are all distinct. If hd_metric is set to 'precomputed', then X_hds must be a 2-D numpy array of floats with shape (N,N) containing the pairwise distances between the data points. This matrix is assumed to be symmetric.
        - labels: 1-D numpy array with N elements, containing integers indicating the class labels of the data points.
        - theta: treshold on the probability mass, around each HD datum, which lies on neighbors with the same class, to fit the precisions of the HD Gaussian neighborhoods. See [1] for further details. This parameter must range in [0.5,1[.
        - init: specify the initialization of the LDS. It is either equal to 'ran', in which case the LD coordinates of the data points are initialized randomly using a Gaussian distribution centered around the origin and with a small variance, or to 'pca', in which case the LD coordinates of the data points are initialized using the PCA projection of the HD samples, or to a 2-D numpy array with N rows, in which case the initial LD coordinates of the data points are specified in the rows of init. In case hd_metric is set to 'precomputed', init can not be set to 'pca'.
        - dim_lds: dimension of the LDS. Must be an integer strictly greater than 0. In case init is a 2-D array, dim_lds must be equal to init.shape[1].
        - nit_max: integer strictly greater than 0 wich specifies the maximum number of gradient descent iterations.
        - rand_state: instance of numpy.random.RandomState. If None, set to numpy.random.
        - hd_metric: metric to compute the HD distances. It must be one of the following:
        --- a string. In this case, it must be one of the following:
        ------ a valid value for the 'metric' parameter of the scipy.spatial.distance.pdist function.
        ------ 'precomputed', in which case X_hds must be a 2-D numpy array of floats with shape (N,N) containing the symmetric pairwise distances between the data points. init must, in this case, be different from 'pca'.
        --- a callable. In this case, it must take two rows of X_hds as parameters and return the distance between the corresponding data points. The distance function is assumed to be symmetric.
        - D_hd_metric: optional dictionary to specify additional arguments to scipy.spatial.distance.pdist, depending on the employed metric.
        - gtol: tolerance on the infinite norm of the gradient during the gradient descent.
        - ftol: tolerance on the relative updates of the objective function during the gradient descent.
        - eef: early exageration factor.
        - eei: number of gradient descent steps to perform with early exageration.
        - mom_init: initial momentum factor value in the gradient descent.
        - mom_fin: final momentum factor value in the gradient descent.
        - mom_t: iteration at which the momentum factor value changes during the gradient descent.
        Out:
        A tuple with:
        - a 2-D numpy array of floats with shape (N, dim_lds), containing the LD representations of the data points in its rows.
        - a 1-D numpy array of floats with N elements. Element at index i indicates the probability mass around X_hds[i,:] which lies on neighbors of the same class.
        """
        # Number of data points
        N = X_hds.shape[0]
        # Checking theta
        if (theta < 0.5) or (theta >= 1):
            raise ValueError("Error in function catsne: theta={theta} whereas it must range in [0.5,1[.".format(theta=theta))
        # Checking rand_state
        if rand_state is None:
            rand_state = np.random
        # Checking init and initializing the LDS
        if isinstance(init, str):
            if init == 'ran':
                X_lds = (10.0**(-4))*rand_state.randn(N, dim_lds)
            elif init == 'pca':
                if isinstance(hd_metric, str) and (hd_metric == "precomputed"):
                    raise ValueError("Error in function catsne: init cannot be set to 'pca' when hd_metric is set to 'precomputed'.")
                from sklearn.decomposition import PCA
                X_lds = PCA(n_components=dim_lds, copy=True, random_state=rand_state).fit_transform(X_hds)
            else:
                raise ValueError("Error in function catsne: init={init} whereas it must either be equal to 'ran' or to 'pca'.".format(init=init))
        else:
            # init must be a 2-D numpy array with N rows and dim_lds columns
            if init.ndim != 2:
                raise ValueError("Error in function catsne: init.ndim={v} whereas init must be a 2-D numpy array.".format(v=init.ndim))
            if init.shape[0] != N:
                raise ValueError("Error in function catsne: init.shape[0]={v} whereas it must equal N={N}.".format(v=init.shape[0], N=N))
            if init.shape[1] != dim_lds:
                raise ValueError("Error in function catsne: init.shape[1]={v} whereas it must equal dim_lds={dim_lds}.".format(v=init.shape[1], dim_lds=dim_lds))
            X_lds = init
        # Computing the squared HD distances
        if isinstance(hd_metric, str):
            if hd_metric == "precomputed":
                ds_hd = X_hds**2.0
            else:
                if D_hd_metric is None:
                    D_hd_metric = {}
                ds_hd = squareform(X=pdist(X=X_hds, metric=hd_metric, **D_hd_metric), force='tomatrix')**2.0
        else:
            # hd_metric is a callable
            ds_hd = np.empty(shape=(N,N), dtype=np.float64)
            for i in range(N):
                ds_hd[i,i] = 0.0
                for j in range(i):
                    ds_hd[i,j] = hd_metric(X_hds[i,:], X_hds[j,:])**2.0
                    ds_hd[j,i] = ds_hd[i,j]
        # Small float
        n_eps = np.finfo(dtype=np.float64).eps
        # Performing momentum gradient descent and returning
        return self.catsne_mgd(progress_listener, show_progress, ds_hd=ds_hd, labels=labels, theta=theta, n_eps=n_eps, eei=eei, eef=eef, X_lds=X_lds, ftol=ftol, N=N, dim_lds=dim_lds, mom_t=mom_t, nit_max=nit_max, gtol=gtol, mom_init=mom_init, mom_fin=mom_fin)


    def catsne_mgd(self, progress_listener, show_progress, ds_hd, labels, theta, n_eps, eei, eef, X_lds, ftol, N, dim_lds, mom_t, nit_max, gtol, mom_init, mom_fin):
        """
        Performing momentum gradient descent in cat-SNE.
        In:
        - ds_hd: 2-D numpy array of floats with shape (N, N), where N is the number of data points. Element [i,j] must be the squared HD distance between data points i and j.
        - labels, theta, eei, eef, ftol, dim_lds, mom_t, nit_max, gtol, mom_init, mom_fin: as in catsne function.
        - n_eps: a small float to avoid making divisions with a denominator close to 0.
        - X_lds: 2-D numpy array of floats with N rows. It contains one example per row and one feature per column. It stores the initial LD coordinates.
        - N: number of data points.
        Out:
        A tuple with:
        - a 2-D numpy array of floats with shape (N, dim_lds), containing the LD representations of the data points in its rows.
        - a 1-D numpy array of floats with N elements. Element at index i indicates the probability mass around X_hds[i,:] which lies on neighbors of the same class.
        """
        # Computing the HD similarities.
        sigma_ijt, max_ti = catsne_hd_sim(ds_hd=ds_hd, labels=labels, theta=theta, n_eps=n_eps)
        # Current number of gradient descent iterations.
        nit = 0
        # Early exageration
        if eei > nit:
            sigma_ijt *= eef
        # Computing the current gradient and objective function values.
        grad, obj = catsne_g(X_lds=X_lds, sigma_ijt=sigma_ijt, nit=nit, eei=eei, eef=eef, n_eps=n_eps)
        gradn = compute_gradn(grad=grad)
        # LD coordinates achieving the smallest value of the objective function.
        best_X_lds = X_lds.copy()
        # Smallest value of the objective function.
        best_obj = obj
        # Objective function value at previous iteration.
        prev_obj = (1+100*ftol)*obj
        rel_obj_diff = compute_rel_obj_diff(prev_obj=prev_obj, obj=obj, n_eps=n_eps)
        # Step size parameters. The steps are adapted during the gradient descent as in [6], using the Delta-Bar-Delta learning rule from [7].
        epsilon, kappa, phi, tdb = 500, 0.2, 0.8, 0.5
        stepsize, delta_bar = epsilon*np.ones(shape=(N, dim_lds), dtype=np.float64), np.zeros(shape=(N, dim_lds), dtype=np.float64)
        # Update of X_lds
        up_X_lds = np.zeros(shape=(N, dim_lds), dtype=np.float64)
        # Gradient descent.
        while (nit <= eei) or (nit <= mom_t) or ((nit < nit_max) and (gradn > gtol) and (rel_obj_diff > ftol)):
            # Computing the step sizes, following the delta-bar-delta rule, from [7].
            delta_bar, stepsize = dbd_rule(delta_bar=delta_bar, grad=grad, stepsize=stepsize, kappa=kappa, phi=phi, tdb=tdb)
            # Performing the gradient descent step with momentum.
            X_lds, up_X_lds = mgd_step(X=X_lds, up_X=up_X_lds, nit=nit, mom_t=mom_t, mom_init=mom_init, mom_fin=mom_fin, stepsize=stepsize, grad=grad)
            # Centering the result
            X_lds -= X_lds.mean(axis=0)
            # Incrementing the iteration counter
            nit += 1
            # Checking whether early exageration is over
            if nit == eei:
                sigma_ijt /= eef
            # Updating the previous objective function value
            prev_obj = obj
            # Computing the gradient at the current LD coordinates and the current objective function value.
            grad, obj = catsne_g(X_lds=X_lds, sigma_ijt=sigma_ijt, nit=nit, eei=eei, eef=eef, n_eps=n_eps)
            gradn = compute_gradn(grad=grad)
            rel_obj_diff = compute_rel_obj_diff(prev_obj=prev_obj, obj=obj, n_eps=n_eps)
            # Updating best_obj and best_X_lds
            if best_obj > obj:
                best_obj, best_X_lds = obj, X_lds.copy()

            if show_progress:
                progress_listener.notify((self.dataset_name, self.proj_name, best_X_lds, self), [])
        # Returning
        return best_X_lds, max_ti

@numba.jit(nopython=True)
def catsne_hd_sim(ds_hd, labels, theta, n_eps):
    """
    Compute the symmetrized HD similarities of cat-SNE, as defined in [1].
    In:
    - ds_hd: 2-D numpy array of floats with shape (N, N), where N is the number of data points. Element [i,j] must be the squared HD distance between data points i and j.
    - labels, theta: see catsne function.
    - n_eps: should be equal to np.finfo(dtype=np.float64).eps.
    Out:
    A tuple with:
    - A 2-D numpy array of floats with shape (N, N) and in which element [i,j] is the symmetrized HD similarity between data points i and j, as defined in [1].
    - A 1-D numpy array of floats with N elements. Element i indicates the probability mass associated to data points with the same class as i in the HD Gaussian neighborhood around i.
    """
    # Number of data points
    N = ds_hd.shape[0]
    # Computing the N**2 HD similarities
    sigma_ij = np.empty(shape=(N,N), dtype=np.float64)
    L = int(round(np.log2(np.float64(N)/2.0)))
    log_perp = np.log(2.0**(np.linspace(L, 1, L).astype(np.float64)))
    max_ti = np.empty(shape=N, dtype=np.float64)
    for i in range(N):
        vi = 1.0
        h = 0
        go = True
        max_ti[i] = -1.0
        labi = labels == labels[i]
        labi[i] = False
        while go and (h < L):
            vi = sne_bs(dsi=ds_hd[i,:], i=i, log_perp=log_perp[h], x0=vi)
            si = sne_sim(dsi=ds_hd[i,:], vi=vi, i=i, compute_log=False)[0]
            h += 1
            ssi = np.sum(si[labi])
            if ssi > max_ti[i]:
                max_ti[i] = ssi
                sigma_ij[i,:] = si
                if max_ti[i] > theta:
                    go = False
    # Symmetrized version
    sigma_ij += sigma_ij.T
    # Returning the normalization of sigma_ij, and max_ti.
    return sigma_ij/np.maximum(n_eps, sigma_ij.sum()), max_ti


@numba.jit(nopython=True)
def sne_bs(dsi, i, log_perp, x0=1.0):
    """
    Binary search to find the root of sne_bsf over vi.
    In:
    - dsi, i, log_perp: same as in sne_bsf function.
    - x0: starting point for the binary search. Must be strictly positive.
    Out:
    A strictly positive float vi such that sne_bsf(dsi, vi, i, log_perp) is close to zero.
    """
    fx0 = sne_bsf(dsi=dsi, vi=x0, i=i, log_perp=log_perp)
    if close_to_zero(v=fx0):
        return x0
    elif not np.isfinite(fx0):
        raise ValueError("Error in function sne_bs: fx0 is nan.")
    elif fx0 > 0:
        x_up, x_low = x0, x0/2.0
        fx_low = sne_bsf(dsi=dsi, vi=x_low, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_low):
            return x_low
        elif not np.isfinite(fx_low):
            # WARNING: can not find a valid root!
            return x_up
        while fx_low > 0:
            x_up, x_low = x_low, x_low/2.0
            fx_low = sne_bsf(dsi=dsi, vi=x_low, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_low):
                return x_low
            if not np.isfinite(fx_low):
                return x_up
    else:
        x_up, x_low = x0*2.0, x0
        fx_up = sne_bsf(dsi=dsi, vi=x_up, i=i, log_perp=log_perp)
        if close_to_zero(v=fx_up):
            return x_up
        elif not np.isfinite(fx_up):
            return x_low
        while fx_up < 0:
            x_up, x_low = 2.0*x_up, x_up
            fx_up = sne_bsf(dsi=dsi, vi=x_up, i=i, log_perp=log_perp)
            if close_to_zero(v=fx_up):
                return x_up
    while True:
        x = (x_up+x_low)/2.0
        fx = sne_bsf(dsi=dsi, vi=x, i=i, log_perp=log_perp)
        if close_to_zero(v=fx):
            return x
        elif fx > 0:
            x_up = x
        else:
            x_low = x

@numba.jit(nopython=True)
def sne_bsf(dsi, vi, i, log_perp):
    """
    Function on which a binary search is performed to find the HD bandwidth of the i^th data point in SNE.
    In:
    - dsi, vi, i: same as in sne_sim function.
    - log_perp: logarithm of the targeted perplexity.
    Out:
    A float corresponding to the current value of the entropy of the similarities with respect to i, minus log_perp.
    """
    si, log_si = sne_sim(dsi=dsi, vi=vi, i=i, compute_log=True)
    return -np.dot(si, log_si) - log_perp


@numba.jit(nopython=True)
def sne_sim(dsi, vi, i, compute_log=True):
    """
    Compute the SNE asymmetric similarities, as well as their log.
    N refers to the number of data points.
    In:
    - dsi: numpy 1-D array of floats with N squared distances with respect to data point i. Element k is the squared distance between data points k and i.
    - vi: bandwidth of the exponentials in the similarities with respect to i.
    - i: index of the data point with respect to which the similarities are computed, between 0 and N-1.
    - compute_log: boolean. If True, the logarithms of the similarities are also computed, and otherwise not.
    Out:
    A tuple with two elements:
    - A 1-D numpy array of floats with N elements. Element k is the SNE similarity between data points i and k.
    - If compute_log is True, a 1-D numpy array of floats with N element. Element k is the log of the SNE similarity between data points i and k. By convention, element i is set to 0. If compute_log is False, it is set to np.empty(shape=N, dtype=np.float64).
    """
    N = dsi.size
    si = np.empty(shape=N, dtype=np.float64)
    si[i] = 0.0
    log_si = np.empty(shape=N, dtype=np.float64)
    indj = arange_except_i(N=N, i=i)
    dsij = dsi[indj]
    log_num_sij = (dsij.min()-dsij)/vi
    si[indj] = np.exp(log_num_sij)
    den_si = si.sum()
    si /= den_si
    if compute_log:
        log_si[i] = 0.0
        log_si[indj] = log_num_sij - np.log(den_si)
    return si, log_si

@numba.jit(nopython=True)
def arange_except_i(N, i):
    """
    Create a 1-D numpy array of integers from 0 to N-1 with step 1, except i.
    In:
    - N: a strictly positive integer.
    - i: a positive integer which is strictly smaller than N.
    Out:
    A 1-D numpy array of integers from 0 to N-1 with step 1, except i.
    """
    arr = np.arange(N)
    return np.hstack((arr[:i], arr[i+1:]))

@numba.jit(nopython=True)
def close_to_zero(v):
    """
    Check whether v is close to zero or not.
    In:
    - v: a scalar or numpy array.
    Out:
    A boolean or numpy array of boolean of the same shape as v, with True when the entry is close to 0 and False otherwise.
    """
    return np.absolute(v) <= 10.0**(-8.0)


def catsne_g(X_lds, sigma_ijt, nit, eei, eef, n_eps):
    """
    Compute the gradient of the objective function of cat-SNE at some LD coordinates, as well as the current value of the objective function.
    In:
    - X_lds: 2-D numpy array of floats with N rows, where N is the number of data points. It contains one example per row and one feature per column. It stores the current LD coordinates.
    - sigma_ijt: 2-D numpy array of floats with shape (N, N), where element [i,j] contains the HD similarity between data points i and j.
    - nit: number of gradient descent steps which have already been performed.
    - eei: number of gradient steps to perform with early exageration.
    - eef: early exageration factor.
    - n_eps: a small float to avoid making divisions with a denominator close to 0.
    Out:
    A tuple with two elements:
    - grad: a 2-D numpy array of floats with the same shape as X_lds, containing the gradient at X_lds.
    - obj: objective function value at X_lds.
    """
    # Computing the LD similarities.
    s_ijt, log_s_ijt, idsld = catsne_ld_sim(ds_ld=squareform(X=pdist(X=X_lds, metric='sqeuclidean'), force='tomatrix'), n_eps=n_eps)

    # Computing the current objective function value
    if nit < eei:
        obj = catsne_obj(sigma_ijt=sigma_ijt/eef, log_s_ijt=log_s_ijt)
    else:
        obj = catsne_obj(sigma_ijt=sigma_ijt, log_s_ijt=log_s_ijt)
    # Computing the gradient.
    c_ij = 4*(sigma_ijt-s_ijt)*idsld
    grad = (X_lds.T*c_ij.dot(np.ones(shape=X_lds.shape[0]))).T - c_ij.dot(X_lds)
    # Returning
    return grad, obj


@numba.jit(nopython=True)
def catsne_obj(sigma_ijt, log_s_ijt):
    """
    Compute the cat-SNE objective function.
    In:
    - sigma_ijt: 2-D numpy array of floats, in which element [i,j] contains the HD similarity between data points i and j, as defined in [1].
    - log_s_ijt: 2-D numpy array of floats, in which element [i,j] contains the log of the LD similarity between data points i and j, as defined in [1].
    Out:
    The value of the cat-SNE objective function.
    """
    return -(sigma_ijt.ravel()).dot(log_s_ijt.ravel())

@numba.jit(nopython=True)
def catsne_ld_sim(ds_ld, n_eps):
    """
    Compute the LD similarities of cat-SNE, as well as their log, as defined in [1].
    In:
    - ds_ld: 2-D numpy array of floats with shape (N, N), where N is the number of data points. Element [i,j] must be the squared LD distance between data points i and j.
    - n_eps: same as in catsne_g function.
    Out:
    A tuple with three elements:
    - A 2-D numpy array of floats with shape (N, N) and in which element [i,j] is the LD similarity between data points i and j.
    - A 2-D numpy array of floats with shape (N, N) and in which element [i,j] is the log of the LD similarity between data points i and j. By convention, the log of 0 is set to 0.
    - 1.0/(1.0+ds_ld)
    """
    ds_ldp = 1.0+ds_ld
    idsld = 1.0/np.maximum(n_eps, ds_ldp)
    s_ijt = idsld.copy()
    log_s_ijt = -np.log(ds_ldp)
    s_ijt = fill_diago(M=s_ijt, v=0.0)
    log_s_ijt = fill_diago(M=log_s_ijt, v=0.0)
    den_s_ijt = s_ijt.sum()
    s_ijt /= np.maximum(n_eps, den_s_ijt)
    log_s_ijt -= np.log(den_s_ijt)
    return s_ijt, log_s_ijt, idsld

@numba.jit(nopython=True)
def fill_diago(M, v):
    """
    Replace the elements on the diagonal of a square matrix M with some value v.
    In:
    - M: a 2-D numpy array storing a square matrix.
    - v: some value.
    Out:
    M, but in which the diagonal elements have been replaced with v.
    """
    for i in range(M.shape[0]):
        M[i,i] = v
    return M

@numba.jit(nopython=True)
def compute_gradn(grad):
    """
    Compute the norm of a gradient.
    In:
    - grad: numpy array of float storing a gradient.
    Out:
    Infinite norm of the gradient.
    """
    return np.absolute(grad).max()


@numba.jit(nopython=True)
def compute_rel_obj_diff(prev_obj, obj, n_eps):
    """
    Compute the relative objective function difference between two steps in a gradient descent.
    In:
    - prev_obj: objective function value at previous iteration.
    - obj: current objective function value.
    - n_eps: a small float that should be equal to np.finfo(dtype=np.float64).eps.
    Out:
    np.abs(prev_obj - obj)/max(np.abs(prev_obj), np.abs(obj))
    """
    return np.abs(prev_obj - obj)/np.maximum(n_eps, max(np.abs(prev_obj), np.abs(obj)))



def dbd_rule(delta_bar, grad, stepsize, kappa=0.2, phi=0.8, tdb=0.5):
    """
    Delta-bar-delta stepsize adaptation rule in a gradient descent procedure, as proposed in [7].
    In:
    - delta_bar: numpy array which stores the current value of the delta bar.
    - grad: numpy array which stores the value of the gradient at the current coordinates.
    - stepsize: numpy array which stores the current values of the step sizes associated with the variables.
    - kappa: linear stepsize increase when delta_bar and the gradient are of the same sign.
    - phi: exponential stepsize decrease when delta_bar and the gradient are of different signs.
    - tdb: parameter for the update of delta_bar.
    Out:
    A tuple with two elements:
    - A numpy array with the update of delta_bar.
    - A numpy array with the update of stepsize.
    """
    dbdp = np.sign(delta_bar) * np.sign(grad)
    stepsize[dbdp > 0] += kappa
    stepsize[dbdp < 0] *= phi
    delta_bar = (1-tdb) * grad + tdb * delta_bar
    return delta_bar, stepsize

@numba.jit(nopython=True)
def mgd_step(X, up_X, nit, mom_t, mom_init, mom_fin, stepsize, grad):
    """
    Momentum gradient descent step.
    In:
    - X: numpy array containing the current value of the variables.
    - up_X: numpy array with the same shape as X storing the update made on the variables at the previous gradient step.
    - nit: number of gradient descent iterations which have already been performed.
    - mom_t: number of gradient descent steps to perform before changing the momentum coefficient.
    - mom_init: momentum coefficient to use when nit<mom_t.
    - mom_fin: momentum coefficient to use when nit>=mom_t.
    - stepsize: step size to use in the gradient descent. Either a scalar, or a numpy array with the same shape as X.
    - grad: numpy array with the same shape as X, storing the gradient of the objective function at the current coordinates.
    Out:
    A tuple with two elements:
    - A numpy array with the updated coordinates, after having performed the momentum gradient descent step.
    - A numpy array storing the update performed on the variables.
    """
    if nit < mom_t:
        mom = mom_init
    else:
        mom = mom_fin
    up_X = mom * up_X - (1-mom) * stepsize * grad
    X += up_X
    return X, up_X
