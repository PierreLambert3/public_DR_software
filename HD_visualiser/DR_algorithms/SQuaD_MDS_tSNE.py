from DR_algorithms.DR_algorithm import DR_algorithm
from DR_algorithms.additional_DR_files.quartet_grads import compute_quartet_grads,compute_quartet_grads_and_LD_distances
import numba, time
import numpy as np
from sklearn.manifold import _barnes_hut_tsne
from sklearn.manifold._t_sne import _joint_probabilities_nn
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KDTree
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from numpy.linalg import norm



def pca(X, m=2):
    return PCA(n_components=m).fit_transform(X)

def select_SVC(C1, C2):
    Xtemp = np.vstack((C1,C2))
    Ytemp = np.vstack((np.zeros((C1.shape[0],1)).astype(int),np.ones((C2.shape[0],1)).astype(int))).ravel()
    # PC = pca(Xtemp, m=min(M-2, 5))
    # mm = LinearSVC(C=0.01, penalty="l1", dual=False).fit(Xtemp, Ytemp)
    mm = ExtraTreesClassifier(n_estimators=50).fit(Xtemp, Ytemp)
    model = SelectFromModel(mm, prefit=True)
    feature_mask = model.get_support()
    return np.where(feature_mask == True)[0]

class SQuaD_MDS_tSNE(DR_algorithm):
    def __init__(self, algo_name):
        super(SQuaD_MDS_tSNE, self).__init__(algo_name)
        self.add_float_hyperparameter('tsne LR multiplier', 0., 10., 0.1, 1.)
        self.add_float_hyperparameter('decay to', 1e-4, 0.1, 1e-4, 1e-4)
        self.add_float_hyperparameter('tSNE exa', 1., 2., 0.01, 2.)
        self.add_bool_hyperparameter('barnes-hut', True)
        self.add_bool_hyperparameter('AUTO LR', True)
        self.add_float_hyperparameter('tSNE LR', 0, 12000, 1, 100)
        self.add_float_hyperparameter('MDS LR', 0, 1000, 1, 50)
        self.add_int_hyperparameter('n iter', 100, 6000, 100, 2000)
        self.add_bool_hyperparameter('nesterov', True)
        self.add_bool_hyperparameter('tsne stop exa', True)
        self.add_bool_hyperparameter('multiple selections', False)
        self.add_bool_hyperparameter('exaggerate D', False)
        self.add_float_hyperparameter('stop exaggeration', 0.4, 1., 0.05, 0.7)
        self.add_bool_hyperparameter('momentum', True)
        self.add_float_hyperparameter('tsne PP', 2., 300., 1., 50.)
        self.add_int_or_string_hyperparameter('on N PC', 3, 50, 1,  "all")
        self.add_bool_hyperparameter('decay LR', True)
        self.add_bool_hyperparameter('show progress', True)
        self.embedding = None


    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")
        hparams = self.get_hyperparameters()

        from sklearn.decomposition import PCA
        Xld = PCA(n_components=2, whiten=True, copy=True).fit_transform(X).astype(np.float64)
        Xld *= 10/np.std(Xld)

        if hparams['on N PC'] != "all":
            N_PC = min(X.shape[1]-1, hparams['on N PC'])
            Xhd_PC = PCA(n_components=N_PC, whiten=True, copy=True).fit_transform(X).astype(np.float64)
            Xld = self.run_SQuaD_MDS_tsne(hparams,Y=Y, Xhd=Xhd_PC, Xld=Xld, progress_listener=progress_listener, momentum=hparams["momentum"])
        else:
            # Xld = self.run_SQuaD_MDS_tsne(hparams, Y=Y, Xhd=X, Xld=Xld, progress_listener=progress_listener, momentum=hparams["momentum"])
            Xld = self.run_hybrid(X, hparams, progress_listener)
        self.embedding = Xld


    def transform(self, X, Y):
        return self.embedding


    def run_hybrid(self, Xhd, hparams, progress_listener):
        N, M = Xhd.shape
        from DR_algorithms.final_squadmds.hybrid import run_hybrid
        dict = {'n iter':hparams['n iter'], 'tSNE exa':hparams['tSNE exa'], 'barnes-hut':hparams['barnes-hut']}
        return run_hybrid(Xhd, dict, progress_stuff=(progress_listener, self))


#         populate_hparams(hparams, N) # set the missing hyperparameters with their default values
#
#
#         '''
#         if the HD data has A LOT of dimensions, using their principal components can speed up the optimisation for a negligible cost given a number of PC sufficiently high and an edaquate intrinsic dimensionality
#         '''
#         if int(hparams['on N PC']) < Xhd.shape[1] and int(hparams['on N PC']) > 1:
#             Xhd = PCA(n_components=int(hparams['on N PC']), whiten=True, copy=True).fit_transform(Xhd).astype(np.float64)
#
#
#         '''
#         init Xld with PCA, then set standard dev of the initialisation to 10
#         why stdev to 10?  ===>  the default learning rate values were found using this scale. Also tSNE often tends to increase the scale during the optimisation, starting at 10 and finishing at 50 is not a problem, whereas starting at 1e-4 and finishing at 40 is a big change in magnitude
#         '''
#         Xld = init_embedding(Xhd)
#
#
#         '''
#         compute HD similarities for tSNE
#         '''
#         perplexities_list = hparams['tSNE perplexities']
#         _P_ = joint_P(Xhd, perplexities_list[0], 'qsdsqd', 1)
#         if len(perplexities_list) > 1:
#             for PP in perplexities_list[1:]:
#                 _P_ += joint_P(Xhd, PP, 'qsdsqd', 1)
#             _P_ /= len(perplexities_list)
#         _P_ *= hparams['tSNE exa']
#
#         # hparams["tsne exa"] = hparams['tSNE exa']
#         # hparams['tsne PP'] = perplexities_list[0]
#         # _P_ = compute_P(Xhd, hparams)
#         # hparams['tsne PP'] = perplexities_list[1]
#         # _P_ = 0.5*_P_ + 0.5*compute_P(Xhd, hparams)
#
#         '''
#         some optimiser params, and memory allocations
#         '''
#         LR_MDS  = hparams['MDS LR']
#         LR_tSNE = hparams['tSNE LR']
#         N_iter = hparams['n iter']
#         tsne_exa_stop = min(500, int(0.3*N_iter))
#         decay = np.exp(np.log(hparams["decay LR to"]) / (N_iter-tsne_exa_stop))
#         # overall_multiplier = 1. # this decays towards hparams["decay LR to"]
#
#
#         momentums      = np.zeros((N, 2))
#         tsne_grads     = np.zeros((N, 2), dtype=np.float32)
#         mds_grads      = np.zeros((N, 2), dtype=np.float32)
#         perms         = np.arange(N)
#         batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
#         Dhd_quartet   = np.zeros((6,))
#         temp_params   = np.zeros_like(Xld)
#
#         prev_t = time.time()
#         for i in range(N_iter):
#             np.random.shuffle(perms) # used for the random quartet designation
#             if i == tsne_exa_stop:  # stop tSNE early exaggeration
#                 _P_ /= hparams['tSNE exa']
#             elif i > tsne_exa_stop: # once early exaggeration is done: start decaying
#                 LR_MDS *= decay
#                 LR_tSNE *= decay
#
#             modified_nesterov2(LR_MDS, LR_tSNE, _P_ , Xld, perms, batches_idxes, mds_grads, tsne_grads, momentums, Xhd, Dhd_quartet, N)
#
#             # for Y. Nestrov's momentums
#             # temp_params = Xld + 0.9*momentums
#
#
#             # # compute both gradients independantly using temp_params
#             # fill_MDS_grads(temp_params, mds_grads, perms, batches_idxes, Xhd, Dhd_quartet)
#             # tsne_grads = KL_divergeance_BH(temp_params.ravel(), _P_, 1, N, 2, 0, False, grad=tsne_grads)
#             #
#             # # don't forget to normalise the gradients befor combining them
#             # mds_grads  /= np.linalg.norm(mds_grads,  axis=0, keepdims=True)
#             # tsne_grads /= np.linalg.norm(tsne_grads, axis=0, keepdims=True)
#             # hybrid_grads = overall_multiplier * (LR_MDS*mds_grads + LR_tSNE*tsne_grads)
#             # # hybrid_grads = overall_multiplier * (LR_tSNE*tsne_grads)
#             # # hybrid_grads = overall_multiplier * (LR_MDS*mds_grads)
#             #
#             # # finally, update the parameters and momentums
#             # momentums = 0.9*momentums - hybrid_grads
#             # Xld += momentums
#             # momentums *= 0.9 # this helps a bit
#
#
#             if time.time() - prev_t > 0.05:
#                 progress_listener.notify((self.dataset_name, self.proj_name, Xld, self), [])
#                 time.sleep(0.05)
#                 prev_t = time.time()
#
#         return Xld

    def run_SQuaD_MDS_tsne(self, hparams, Y, Xhd, Xld, progress_listener = None, momentum = False):
        prev_t = time.time()
        N, M = Xhd.shape
        show_progress = (None not in [progress_listener, self.dataset_name, self.proj_name]) and hparams["show progress"]
        n_iter = hparams["n iter"]
        nesterov  = hparams["nesterov"]
        if hparams['AUTO LR'] == True:
            LR    = 0.06 * N
            LRmds = LR / 2
        else:
            LRmds     = hparams["MDS LR"]
            LR     = hparams["tSNE LR"]


        if hparams["tsne stop exa"]:
            tsne_exa_stop      = min(300, int(0.3*n_iter))
        else:
            tsne_exa_stop = 99999999999
        if hparams['decay LR']:
            decay = np.exp(np.log(hparams["decay to"]) / (n_iter-tsne_exa_stop))
        else:
            decay = 1.
        # exaggeration of HD distances by taking them squared
        squared_D  = False
        stop_D_exa = 0
        if hparams["exaggerate D"]: # exaggeration of HD distances by taking them squared
            stop_D_exa = int(n_iter*hparams["stop exaggeration"]) # iteration when we stop the exaggeration
            squared_D  = True

        # build P matrix
        P_ = compute_P(Xhd, hparams)
        tmp = hparams['tsne PP']
        # hparams['tsne PP'] = 5.
        hparams['tsne PP'] = 4.
        P_ = 0.5*P_ + 0.5*compute_P(Xhd, hparams)
        hparams['tsne PP'] = tmp


        # for i in range(400):
        #     print("hparam ==> PP=45 et n_iter =2000    ===> as good as fast MS with good hparams")
        # hparam ==> PP=45 et n_iter =2000
        # ===> as good as fast MS with good hparams

        closest_points        = np.zeros((N, 5), dtype=np.int32)
        closest_dists         = np.zeros((N, 5), dtype=np.float64)
        argmin_closest_points = np.zeros((N,), dtype=np.int32)
        momentums      = np.zeros((N, 2))
        momentums_both      = np.zeros((N, 2))
        momentums_mds = np.zeros((N, 2))
        momentums_tsne = np.zeros((N, 2))
        grad_momentums      = np.zeros((N, 2))
        tsne_grad = np.zeros((N, 2), dtype=np.float32)
        mds_grad = np.zeros((N, 2))
        grad_momentums_tsne = np.zeros((N, 2))
        grad_momentums_EMA = np.zeros((N, 2))
        absgrad_EMA   = np.ones((N, 2))
        gains         = np.ones((N, 2))
        gains2        = np.ones((N, 2))
        update        = np.zeros((N, 2))
        update2       = np.zeros((N, 2))
        tmp_acc       = np.ones((N, 2))
        grad_acc      = np.ones((N, 2))
        Ys = Xld.copy()
        Zs = Xld.copy()
        A_k = 0.
        change = np.zeros_like(Xld)


        # print(decay, tsne_exa_stop, n_iter)
        # 1/0

        for ww in range(1): # NORMAL VERSION
        # for ww in range(4):
            # choices = np.random.choice(np.arange(len(interesting_idxs)), replace=False, size=2)
            # C1 = Xhd[interesting_idxs[choices[0]]]
            # C2 = Xhd[interesting_idxs[choices[1]]]



            # X_hd_MDS = Xhd.copy()[:, selected_indexes]
            X_hd_MDS = Xhd.copy()   # NORMAL VERSION
            P = P_.copy()

            perms         = np.arange(N)
            batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
            Dhd_quartet   = np.zeros((6,))
            # momentum_alpha = 0.99
            momentum_alpha = 0.99
            mul = 1.
            for i in range(n_iter):

                if i > tsne_exa_stop:
                    mul *= decay
                if i == stop_D_exa:
                    squared_D = False
                if i == tsne_exa_stop:
                    P /= hparams["tsne exa"]

                np.random.shuffle(perms)

                modified_nesterov(LRmds,mul, N, P, Xld, LR, perms, batches_idxes, mds_grad, tsne_grad, momentums, momentums_mds, momentums_tsne, Xhd, squared_D, Dhd_quartet)


                # if show_progress and i % 500 == 0:
                if show_progress and time.time() - prev_t > 0.05:
                    progress_listener.notify((self.dataset_name, self.proj_name, Xld, self), [])
                    time.sleep(0.05)
                    prev_t = time.time()
                # print(np.std(Xld), " Xld std")
        return Xld

def modified_nesterov(LR_mds,decayed_mul, N, P, Xld, LR_tsne, perms, batches_idxes, mds_grad, tsne_grad, momentums, momentums_mds_grad, momentums_tsne_grad, Xhd, squared_D, Dhd_quartet):

    params_temp  = Xld + 0.9*momentums
    mds_grad  = batch_optim(params_temp, perms, batches_idxes, mds_grad, Xhd, squared_D, Dhd_quartet)
    tsne_grad = KL_divergeance_BH(params_temp.ravel(), P, 1, N, 2, 0, False, grad=tsne_grad)

    mds_grad  /= np.linalg.norm(mds_grad, axis=0, keepdims=True)
    tsne_grad /= np.linalg.norm(tsne_grad, axis=0, keepdims=True)


    le_gradient = (LR_mds*mds_grad + LR_tsne*tsne_grad) * decayed_mul
    momentums = 0.9*momentums - le_gradient
    Xld += momentums
    momentums *= 0.9

def modified_nesterov2(LR_mds, LR_tsne, P, Xld, perms, batches_idxes, mds_grads, tsne_grads, momentums, Xhd, Dhd_quartet, N):
    params_temp  = Xld + 0.9*momentums
    mds_grads  = batch_optim(params_temp, perms, batches_idxes, mds_grads, Xhd, False, Dhd_quartet)
    tsne_grads = KL_divergeance_BH(params_temp.ravel(), P, 1, N, 2, 0, False, grad=tsne_grads)

    mds_grads  /= np.linalg.norm(mds_grads, axis=0, keepdims=True)
    tsne_grads /= np.linalg.norm(tsne_grads, axis=0, keepdims=True)


    # le_gradient =
    momentums = 0.9*momentums - (LR_mds*mds_grads + LR_tsne*tsne_grads)
    Xld += momentums
    momentums *= 0.9

    tsne_grads.fill(0.)
    mds_grads.fill(0.)



def fast_distance_scaling_update_tsne_momentum(change, A_k, Ys, Zs, nesterov,absgrad_EMA,closest_points,closest_dists,argmin_closest_points,momentum_alpha, N, P, Xld, LR, perms, batches_idxes, grad_acc, tmp_acc, gains, update,gains2, update2,momentums, momentums_both, grad_momentums, grad_momentums_tsne, grad_momentums_EMA, Xhd, squared_D, Dhd_quartet, tsne_LR_multiplier):
    LR_MDS = 500
    # ---------------------- BACKUP ---------------------------
    if not nesterov:
        movement_before = 1.
        XLD_temp = X_LD.copy()
    else:
        movement_before = (tsne_LR_multiplier*LR*grad_momentums_tsne + 0.1*LR_MDS*grad_momentums/(tsne_LR_multiplier))
        XLD_temp = X_LD - 0.9*movement_before

    # step 1 : distance scaling gradients using the quartet method
    batch_optim_momentum(absgrad_EMA,closest_points,closest_dists,argmin_closest_points,momentum_alpha, XLD_temp, perms, batches_idxes, grad_acc, tmp_acc, gains, update, gains2, update2, grad_momentums, grad_momentums_EMA, Xhd, squared_D, Dhd_quartet)

    # step 2 : t-SNE gradients, the code is taken for scikit-learn's github
    tsne_grad = KL_divergeance_BH(XLD_temp.ravel(), P, 1, N, 2, 0, False)
    # X_LD -= tsne_LR_multiplier*LR*tsne_grad

    grad_momentums_tsne *= momentum_alpha
    grad_momentums_tsne += 1.*tsne_grad


    # X_LD -= 0.3*movement_before + LR * (  tsne_LR_multiplier*grad_momentums_tsne   +   grad_momentums/(10*tsne_LR_multiplier)  )
    X_LD -= 0.9*movement_before +(   LR*tsne_LR_multiplier*grad_momentums_tsne   +   0.1*LR_MDS*grad_momentums/(tsne_LR_multiplier)  )
    # grad_momentums_tsne *= 0.95

    grad_momentums_tsne *= 0.9
    grad_momentums *= 0.95
    # ---------------------- END BACKUP ---------------------------


# aussi tester sans forces repulsives de tsne
#
# puis faire le truc avec supervision groupe A et B  et manifold smoothing sur ANN cf feuille




@numba.jit(nopython=True, fastmath=True)
def batch_optim_momentum(absgrad_EMA,closest_points,closest_dists,argmin_closest_points,momentum_alpha, X_LD, perms, batches_idxes, grad_acc, tmp_acc, gains, update, gains2, update2, momentums, momentums_EMA, Xhd, squared_D, Dhd_quartet):
    # tmp_acc.fill(0.)
    grad_acc.fill(0.)
    # beta = 0.3
    # beta = 0.15
    beta = 1.
    # momentum_alpha = 0.95
    # beta = 0.1
    for batch_idx in batches_idxes:
        quartet     = perms[batch_idx]
        LD_points   = X_LD[quartet]

        # compute quartet's HD distances
        if squared_D: # during exaggeration: dont take the square root of the distances
            Dhd_quartet[0] = np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2)
            Dhd_quartet[1] = np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2)
            Dhd_quartet[2] = np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2)
            Dhd_quartet[3] = np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2)
            Dhd_quartet[4] = np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2)
            Dhd_quartet[5] = np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2)
        else:
            Dhd_quartet[0] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2))
            Dhd_quartet[1] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2))
            Dhd_quartet[2] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2))
            Dhd_quartet[3] = np.sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2))
            Dhd_quartet[4] = np.sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2))
            Dhd_quartet[5] = np.sqrt(np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2))

        Dhd_quartet  /= (np.sum(Dhd_quartet)+1e-9)
        # quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet)
        quartet_grads, Dld_quartet = compute_quartet_grads_and_LD_distances(LD_points, Dhd_quartet)

        momentums[quartet[0], 0] = momentum_alpha*momentums[quartet[0], 0] + beta*quartet_grads[0]
        momentums[quartet[0], 1] = momentum_alpha*momentums[quartet[0], 1] + beta*quartet_grads[1]
        momentums[quartet[1], 0] = momentum_alpha*momentums[quartet[1], 0] + beta*quartet_grads[2]
        momentums[quartet[1], 1] = momentum_alpha*momentums[quartet[1], 1] + beta*quartet_grads[3]
        momentums[quartet[2], 0] = momentum_alpha*momentums[quartet[2], 0] + beta*quartet_grads[4]
        momentums[quartet[2], 1] = momentum_alpha*momentums[quartet[2], 1] + beta*quartet_grads[5]
        momentums[quartet[3], 0] = momentum_alpha*momentums[quartet[3], 0] + beta*quartet_grads[6]
        momentums[quartet[3], 1] = momentum_alpha*momentums[quartet[3], 1] + beta*quartet_grads[7]

        grad_acc[quartet[0], 0] = quartet_grads[0]
        grad_acc[quartet[0], 1] = quartet_grads[1]
        grad_acc[quartet[1], 0] = quartet_grads[2]
        grad_acc[quartet[1], 1] = quartet_grads[3]
        grad_acc[quartet[2], 0] = quartet_grads[4]
        grad_acc[quartet[2], 1] = quartet_grads[5]
        grad_acc[quartet[3], 0] = quartet_grads[6]
        grad_acc[quartet[3], 1] = quartet_grads[7]


@numba.jit(nopython=True, fastmath=True)
def batch_optim(X_LD, perms, batches_idxes, grad_acc, Xhd, squared_D, Dhd_quartet):
    grad_acc.fill(0.)
    for batch_idx in batches_idxes:
        quartet = perms[batch_idx]
        LD_points   = X_LD[quartet]

        # compute quartet's HD distances
        if squared_D: # during exaggeration: dont take the square root of the distances
            Dhd_quartet[0] = np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2)
            Dhd_quartet[1] = np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2)
            Dhd_quartet[2] = np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2)
            Dhd_quartet[3] = np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2)
            Dhd_quartet[4] = np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2)
            Dhd_quartet[5] = np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2)
        else:
            Dhd_quartet[0] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2))
            Dhd_quartet[1] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2))
            Dhd_quartet[2] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2))
            Dhd_quartet[3] = np.sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2))
            Dhd_quartet[4] = np.sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2))
            Dhd_quartet[5] = np.sqrt(np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2))

        Dhd_quartet  /= np.sum(Dhd_quartet)
        quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet)


        grad_acc[quartet[0], 0] += quartet_grads[0]
        grad_acc[quartet[0], 1] += quartet_grads[1]
        grad_acc[quartet[1], 0] += quartet_grads[2]
        grad_acc[quartet[1], 1] += quartet_grads[3]
        grad_acc[quartet[2], 0] += quartet_grads[4]
        grad_acc[quartet[2], 1] += quartet_grads[5]
        grad_acc[quartet[3], 0] += quartet_grads[6]
        grad_acc[quartet[3], 1] += quartet_grads[7]
    return grad_acc

def fast_distance_scaling_update_tsne(N, P, X_LD, LR, perms, batches_idxes, grad_acc, Xhd, squared_D, Dhd_quartet, tsne_LR_multiplier):
    grad_acc.fill(0.)

    # step 1 : distance scaling gradients using the quartet method
    batch_optim(X_LD, perms, batches_idxes, grad_acc, Xhd, squared_D, Dhd_quartet)

    # step 2 : t-SNE gradients, the code is taken for scikit-learn's github
    tsne_grad = KL_divergeance_BH(X_LD.ravel(), P, 1, N, 2, 0, False)

    X_LD -= tsne_LR_multiplier*LR*tsne_grad.reshape((N, 2))
    X_LD -= LR*grad_acc

# t-SNE's joint P matrix, with possible exaggeration
def compute_P(X, hparams):
    P = joint_P(X, hparams["tsne PP"], "barnes_hut", 4)
    if hparams["tsne exa"] > 1.:
        P *= hparams["tsne exa"]
    return P

# t-SNE's joint P matrix
def joint_P(X, PP, method, N_jobs):
    from sklearn.neighbors import NearestNeighbors
    N, dim = X.shape
    n_neighbors = min(N - 1, int(3.*PP + 1))
    knn = NearestNeighbors(algorithm='auto', n_jobs=N_jobs, n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X)
    D_nn = knn.kneighbors_graph(mode='distance')
    D_nn.data **= 2
    del knn
    P = joint_probabilities_nn(D_nn, PP)
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
    P = csr_matrix((conditional_P.ravel(), D.indices, D.indptr), shape=(N, N))
    P = P + P.T
    sum_P = np.maximum(P.sum(), np.finfo(np.double).eps)
    P /= sum_P
    assert np.all(np.abs(P.data) <= 1.0)
    return P

def KL_divergeance_BH(flat_X_LD, P, degrees_of_freedom, n_samples, n_components,
                      skip_num_points, compute_error, grad,
                      angle=0.75, verbose=False,  num_threads=1):
    grad.fill(0.)
    flat_X_LD = flat_X_LD.astype(np.float32, copy=False)
    X_embedded = flat_X_LD.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)
    error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr,
                                      grad, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error,
                                      num_threads=num_threads)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c
    return grad.reshape((n_samples, n_components))
