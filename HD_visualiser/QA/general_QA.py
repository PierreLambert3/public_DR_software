import numpy as np
import numba

'''
the functions coranking(), eval_auc(), eval_dr_quality(), red_rnx_auc(), and eval_rnx()
are taken from Cyril de Bodt 's github page at https://github.com/cdebodt/Fast_Multi-scale_NE
Rnx is introduced in : "Quality assessment of dimensionality reduction: Rank-based criteria"
by : Lee, John and Verleysen, Michel
in: Neurocomputing, 2009
'''
def coranking(d_hd, d_ld):
    # Computing the permutations to sort the rows of the distance matrices in HDS and LDS.
    perm_hd = d_hd.argsort(axis=-1, kind='mergesort')
    perm_ld = d_ld.argsort(axis=-1, kind='mergesort')

    N = d_hd.shape[0]
    i = np.arange(N, dtype=np.int64)

    # Computing the ranks in the LDS
    R = np.empty(shape=(N,N), dtype=np.int64)
    for j in range(N):
        R[perm_ld[j, i],j] = i

    # Computing the co-ranking matrix
    Q = np.zeros(shape=(N,N), dtype=np.int64)
    for j in range(N):
        Q[i,R[perm_hd[j,i],j]] += 1
    # Returning
    return Q[1:,1:]

@numba.jit(nopython=True)
def eval_auc(arr):
    """
    Evaluates the AUC, as defined in [2].
    In:
    - arr: 1-D numpy array storing the values of a curve from K=1 to arr.size.
    Out:
    The AUC under arr, as defined in [2], with a log scale for K=1 to arr.size.
    """
    i_all_k = 1.0/(np.arange(arr.size)+1.0)
    return np.float64(np.dot(arr, i_all_k))/(i_all_k.sum())


def eval_dr_quality(d_hd, d_ld):
    """
    Compute the DR quality assessment criteria R_{NX}(K) and AUC, as defined in [2, 3, 4, 5] and as employed in the experiments reported in [1, 7].
    These criteria measure the neighborhood preservation around the data points from the HDS to the LDS.
    Based on the HD and LD distances, the sets v_i^K (resp. n_i^K) of the K nearest neighbors of data point i in the HDS (resp. LDS) can first be computed.
    Their average normalized agreement develops as Q_{NX}(K) = (1/N) * \sum_{i=1}^{N} |v_i^K \cap n_i^K|/K, where N refers to the number of data points and \cap to the set intersection operator.
    Q_{NX}(K) ranges between 0 and 1; the closer to 1, the better.
    As the expectation of Q_{NX}(K) with random LD coordinates is equal to K/(N-1), which is increasing with K, R_{NX}(K) = ((N-1)*Q_{NX}(K)-K)/(N-1-K) enables more easily comparing different neighborhood sizes K.
    R_{NX}(K) ranges between -1 and 1, but a negative value indicates that the embedding performs worse than random. Therefore, R_{NX}(K) typically lies between 0 and 1.
    The R_{NX}(K) values for K=1 to N-2 can be displayed as a curve with a log scale for K, as closer neighbors typically prevail.
    The area under the resulting curve (AUC) is a scalar score which grows with DR quality, quantified at all scales with an emphasis on small ones.
    The AUC lies between -1 and 1, but a negative value implies performances which are worse than random.
    In:
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    Out: a tuple with
    - a 1-D numpy array with N-2 elements. Element i contains R_{NX}(i+1).
    - the AUC of the R_{NX}(K) curve with a log scale for K, as defined in [2].
    Remark:
    - The time complexity to evaluate the quality criteria is O(N**2 log(N)). It is the time complexity to compute the co-ranking matrix. R_{NX}(K) can then be evaluated for all K=1, ..., N-2 in O(N**2).
    """
    # Computing the co-ranking matrix of the embedding, and the R_{NX}(K) curve.
    rnxk = eval_rnx(Q=coranking(d_hd=d_hd, d_ld=d_ld))
    # Computing the AUC, and returning.
    return rnxk, eval_auc(rnxk)

def red_rnx_auc(X_hds, X_lds, Kup=10000):
    """
    This 'red_rnx_auc' function is similar to 'eval_dr_quality', as it computes the DR quality assessment criteria R_{NX}(K) and AUC, as employed in the experiments of [1], but it evaluates these criteria only for the neighborhood sizes K up to Kup, at the opposite of the 'eval_dr_quality' function which considers all possible neighborhood sizes.
    For a description of the quality criteria and how they can be interpreted, check the documentation of the 'eval_dr_quality' function.
    While the 'eval_dr_quality' function has a O(N**2 log(N)) time complexity when the considered data set has N samples, this 'red_rnx_auc' function has a O(Kup * N * log(N)) time complexity. Provided that Kup is small compared to N, it can hence be employed on much larger databases than 'eval_dr_quality', which is limited to data sets with a few thousands samples.
    At the opposite of the 'eval_dr_quality' function, which can be employed using any types of distances in the HDS and the LDS, this 'red_rnx_auc' function is only considering Euclidean distances, in both the HDS and the LD embedding.
    The R_{NX}(K) values computed by this function can be displayed as a curve for K=1 to Kup, with a log scale for K, as closer neighbors typically prevail.
    The area under the resulting reduced R_{NX} curve (AUC) is a scalar score which grows with DR quality for neighborhood sizes up to Kup.
    R_{NX}(K) ranges between -1 and 1, but a negative value indicates that the embedding performs worse than random. Therefore, R_{NX}(K) typically lies between 0 and 1.
    The AUC lies between -1 and 1, but a negative value implies performances which are worse than random for the neighborhood sizes smaller than Kup.
    In:
    - X_hds: 2-D numpy.ndarray with N rows, containing the HD data set, with one example per row and one feature per column.
    - X_lds: 2-D numpy.ndarray with N rows, containing the LD data set representing X_hds. It contains one example per row and one feature per column. X_lds[i,:] contains the LD coordinates of the HD sample X_hds[i,:]. If X_lds.shape[0] is not equal to X_hds.shape[0], an error is raised.
    - Kup: largest neighborhood size to consider when computing the quality criteria. It must be an integer >= 1 and <= X_hds.shape[0]-1, otherwise an error is raised.
    Out: a tuple with
    - a 1-D numpy array with min(Kup, N-2) elements. Element at index i, starting from 0, contains R_{NX}(i+1).
    - a scalar being the AUC of the R_{NX}(K) curve with a log scale for K, with K ranging from 1 to Kup, as defined in [1].
    Remark:
    - The time complexity of this function is O(Kup*N*log(N)).
    - Euclidean distances are employed in both the HDS and the LDS.
    """
    global module_name
    # Number N of examples in the data set and number of HD dimensions
    N, M = X_hds.shape
    # Number of LD dimensions
    P = X_lds.shape[1]
    # Checking that X_lds also has N rows
    if not np.isclose(N, X_lds.shape[0]):
        raise ValueError("Error in function red_rnx_auc of module {module_name}: X_hds.shape[0]={N} whereas X_lds.shape[0]={M}.".format(module_name=module_name, N=N, M=X_lds.shape[0]))
    # Checking that Kup is an integer >=1 and <= N-1
    if (not isinstance(Kup, int)) or (Kup < 1) or (Kup > N-1):
        raise ValueError("Error in function red_rnx_auc of module {module_name}: Kup={Kup} whereas it should be an integer >= 1 and <={v}.".format(module_name=module_name, v=N-1, Kup=Kup))
    # Initializing the arrays to store the reduced Q_NX and R_NX curves.
    qnx = np.empty(shape=Kup, dtype=np.float64)
    rnx_size = min(Kup, N-2)
    rnx = np.empty(shape=rnx_size, dtype=np.float64)
    # Reshaping X_hds and X_lds
    X_hds_1D = np.ascontiguousarray(a=np.reshape(a=X_hds, newshape=N*M, order='C'), dtype=np.float64)
    X_lds_1D = np.ascontiguousarray(a=np.reshape(a=X_lds, newshape=N*P, order='C'), dtype=np.float64)
    # Computing the reduced quality criteria
    # auc = QA_cython.drqa_qnx_rnx_auc(X_hds_1D, X_lds_1D, N, M, P, Kup, qnx, rnx, rnx_size)
    auc  = cython_implem.drqa_qnx_rnx_auc(X_hds_1D, X_lds_1D, N, M, P, Kup, qnx, rnx, rnx_size)
    # Returning
    return rnx, auc

@numba.jit(nopython=True)
def eval_rnx(Q):
    """
    Evaluate R_NX(K) for K = 1 to N-2, as defined in [5]. N is the number of data points in the data set.
    The time complexity of this function is O(N^2).
    In:
    - Q: a 2-D numpy array representing the (N-1)x(N-1) co-ranking matrix of the embedding.
    Out:
    A 1-D numpy array with N-2 elements. Element i contains R_NX(i+1).
    """
    N_1 = Q.shape[0]
    N = N_1 + 1
    # Computing Q_NX
    qnxk = np.empty(shape=N_1, dtype=np.float64)
    acc_q = 0.0
    for K in range(N_1):
        acc_q += (Q[K,K] + np.sum(Q[K,:K]) + np.sum(Q[:K,K]))
        qnxk[K] = acc_q/((K+1)*N)
    # Computing R_NX
    arr_K = np.arange(N_1)[1:].astype(np.float64)
    rnxk = (N_1*qnxk[:N_1-1]-arr_K)/(N_1-arr_K)
    # Returning
    return rnxk

def local_rnx_auc(dataset_name, proj_name, perms_hd, perms_ld, D_HD, D_LD, KNN_points, listener):
    ratio = perms_ld.shape[0]/KNN_points.shape[0]
    N_1 = perms_ld.shape[0]-1
    Qnx = np.zeros((N_1, ))
    n_repeat = int(D_HD.shape[0]/KNN_points.shape[0])
    remainer = D_HD.shape[0] - KNN_points.shape[0]*n_repeat


    D_HD_local = D_HD[KNN_points]
    D_LD_local = D_LD[KNN_points]
    D_HD_local = np.repeat(D_HD_local, n_repeat, axis=0)
    D_LD_local = np.repeat(D_LD_local, n_repeat, axis=0)
    if remainer > 0:
        r = np.random.randint(D_HD_local.shape[0], size = (remainer,))
        D_HD_local = np.vstack((D_HD_local, D_HD_local[r]))
        D_LD_local = np.vstack((D_LD_local, D_LD_local[r]))

    q = coranking(D_HD_local, D_LD_local)


    rnxk = eval_rnx(q)
    auc  = eval_auc(rnxk)
    listener.notify((dataset_name, proj_name, rnxk, auc), [])
    # Computing the AUC, and returning.
    # return rnxk, eval_auc(rnxk)


'''
knngain() was taken from C. de Bodt's github at https://github.com/cdebodt/cat-SNE/blob/master/catsne.py
KNN gain follows the ide behind Rnx(K) curves but using KNN accuracy. It is defined in:
de Bodt, C., Mulders, D., López-Sánchez, D., Verleysen, M., & Lee, J. A. (2019). Class-aware t-SNE: cat-SNE. In ESANN (pp. 409-414)
'''
@numba.jit(nopython=True)
def knngain(d_hd, d_ld, labels):
    """
    Compute the KNN gain curve and its AUC, as defined in [6].
    If c_i refers to the class label of data point i, v_i^K (resp. n_i^K) to the set of the K nearest neighbors of data point i in the HDS (resp. LDS), and N to the number of data points, the KNN gain develops as G_{NN}(K) = (1/N) * \sum_{i=1}^{N} (|{j \in n_i^K such that c_i=c_j}|-|{j \in v_i^K such that c_i=c_j}|)/K.
    It averages the gain (or loss, if negative) of neighbors of the same class around each point, after DR.
    Hence, a positive value correlates with likely improved KNN classification performances.
    As the R_{NX}(K) curve from the unsupervised DR quality assessment, the KNN gain G_{NN}(K) can be displayed with respect to K, with a log scale for K.
    A global score summarizing the resulting curve is provided by its area (AUC).
    In:
    - d_hd: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the HDS.
    - d_ld: 2-D numpy array of floats with shape (N, N), representing the redundant matrix of pairwise distances in the LDS.
    - labels: 1-D numpy array with N elements, containing integers indicating the class labels of the data points.
    Out:
    A tuple with:
    - a 1-D numpy array of floats with N-1 elements, storing the KNN gain for K=1 to N-1.
    - the AUC of the KNN gain curve, with a log scale for K.
    """
    # Number of data points
    N = d_hd.shape[0]
    N_1 = N-1
    k_hd = np.zeros(shape=N_1, dtype=np.int64)
    k_ld = np.zeros(shape=N_1, dtype=np.int64)
    # For each data point
    for i in range(N):
        c_i = labels[i]
        di_hd = d_hd[i,:].argsort(kind='mergesort')
        di_ld = d_ld[i,:].argsort(kind='mergesort')
        # Making sure that i is first in di_hd and di_ld
        for arr in [di_hd, di_ld]:
            for idj, j in enumerate(arr):
                if j == i:
                    idi = idj
                    break
            if idi != 0:
                arr[idi] = arr[0]
            arr = arr[1:]
        for k in range(N_1):
            if c_i == labels[di_hd[k]]:
                k_hd[k] += 1
            if c_i == labels[di_ld[k]]:
                k_ld[k] += 1
    # Computing the KNN gain
    gn = (k_ld.cumsum() - k_hd.cumsum()).astype(np.float64)/((1.0+np.arange(N_1))*N)
    # Returning the KNN gain and its AUC
    return gn, eval_auc(gn)

@numba.jit(nopython=True)
def knngain_regression(d_hd, d_ld, labels):
    # Number of data points
    N = d_hd.shape[0]
    N_1 = N-1
    k_hd = np.zeros(shape=N_1, dtype=np.int64)
    k_ld = np.zeros(shape=N_1, dtype=np.int64)
    std_label = np.std(labels)
    # For each data point
    for i in range(N):
        c_i = labels[i]
        di_hd = d_hd[i,:].argsort(kind='mergesort')
        di_ld = d_ld[i,:].argsort(kind='mergesort')
        # Making sure that i is first in di_hd and di_ld
        for arr in [di_hd, di_ld]:
            for idj, j in enumerate(arr):
                if j == i:
                    idi = idj
                    break
            if idi != 0:
                arr[idi] = arr[0]
            arr = arr[1:]
        for k in range(N_1):
            k_hd[k] += np.abs(c_i - labels[di_hd[k]]) / std_label
            k_ld[k] += np.abs(c_i - labels[di_ld[k]]) / std_label
    # Computing the KNN gain
    gn = (k_hd.cumsum() - k_ld.cumsum()).astype(np.float64)/((1.0+np.arange(N_1))*N)
    # Returning the KNN gain and its AUC
    return gn, eval_auc(gn)

class generalQA():
    def __init__(self, N):
        self.N = N

        self.Rnx       = np.zeros((N-1,))
        self.Rnx_AUC   = 0.
        self.Rnx_ready = False

        self.Dcorr       = 0.
        self.Dcorr_ready = False

        self.KNNgain       = np.zeros((N-1,))
        self.KNNgain_AUC   = 0.
        self.KNNgain_ready = False
