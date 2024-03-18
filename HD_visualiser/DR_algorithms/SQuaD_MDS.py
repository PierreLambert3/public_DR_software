from DR_algorithms.DR_algorithm import DR_algorithm
from DR_algorithms.additional_DR_files.quartet_grads import compute_quartet_grads
import numba
import numpy as np
from numpy.linalg import norm

'''
TODO: attractive forces = tSNE
replusive = SQUADMDS
'''

class SQuaD_MDS(DR_algorithm):
    def __init__(self, algo_name):
        super(SQuaD_MDS, self).__init__(algo_name)
        self.add_bool_hyperparameter('version 1', False)
        self.add_string_hyperparameter('metric', ["euclidean", "relative rbf distance"], "euclidean")
        self.add_bool_hyperparameter('exaggerate D', False)
        self.add_float_hyperparameter('stop exaggeration', 0.4, 1., 0.05, 0.7)
        self.add_float_hyperparameter('LR', 1, 1000, 1, 550)
        self.add_int_hyperparameter('n iter', 100, 50000, 100, 3000)
        self.add_int_or_string_hyperparameter('MDS on N PC', 3, 50, 1,  "all")
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

        if hparams['MDS on N PC'] != "all":
            N_PC = min(X.shape[1]-1, hparams['MDS on N PC'])
            Xhd_PC = PCA(n_components=N_PC, whiten=True, copy=True).fit_transform(X).astype(np.float64)
            Xld = self.run_SQuaD_MDS(hparams, Xhd=Xhd_PC, Xld=Xld, progress_listener=progress_listener)
        else:
            Xld = self.run_SQuaD_MDS(hparams, Xhd=X, Xld=Xld, progress_listener=progress_listener)
        self.embedding = Xld
        super(SQuaD_MDS, self).compute_stress(X, " SQuaD_MDS ")


    def transform(self, X, Y):
        return self.embedding

    def run_SQuaD_MDS(self, hparams, Xhd, Xld, progress_listener = None):
        if hparams["version 1"]:
            from DR_algorithms.final_squadmds.SQuaD_MDS import run_SQuaD_MDS
            return run_SQuaD_MDS(Xhd, {'n iter': hparams["n iter"], 'LR': hparams["LR"]}, progress_stuff=(progress_listener, self))
        else:
            from DR_algorithms.final_squadmds.SQuaD_MDS import run_SQuaD_MDS_version2
            return run_SQuaD_MDS_version2(Xhd, {'n iter': hparams["n iter"], 'LR': hparams["LR"]}, progress_stuff=(progress_listener, self))

    def run_SQuaD_MDS_old(self, hparams, Xhd, Xld, progress_listener = None):
        N, M = Xhd.shape

        show_progress         = (None not in [progress_listener, self.dataset_name, self.proj_name]) and hparams["show progress"]
        relative_rbf_distance = hparams["metric"] == "relative rbf distance" # transform the distances nonlinearly with 1 - exp(- (Dhd - min(Dhd))/(2*std(Dhd)) ) as described in the paper
        n_iter                = hparams["n iter"]
        LR                    = hparams["LR"]
        decay = np.exp(np.log(1e-3) / n_iter) # aim for an end LR of 1e-3 , if not initialised with a std of 10 (not recommended), then this value should be changed as well
        squared_D  = False
        stop_D_exa = 0
        if hparams["exaggerate D"]: # exaggeration of HD distances by taking them squared
            stop_D_exa = int(n_iter*hparams["stop exaggeration"]) # iteration when we stop the exaggeration
            squared_D  = True

        perms         = np.arange(N)
        batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
        grad_acc      = np.ones((N, 2))
        Dhd_quartet   = np.zeros((6,))
        for i in range(n_iter):
            LR *= decay
            if i == stop_D_exa:
                squared_D = False
            np.random.shuffle(perms)
            fast_distance_scaling_update(N, Xld, LR, perms, batches_idxes, grad_acc, Xhd, squared_D, Dhd_quartet, relative_rbf_distance)
            if show_progress and i % 10 == 0:
                progress_listener.notify((self.dataset_name, self.proj_name, Xld, self), [])

    def run_SQuaD_MDS_momentum(self, hparams, Xhd, Xld, progress_listener = None):
        N, M = Xhd.shape

        show_progress         = (None not in [progress_listener, self.dataset_name, self.proj_name]) and hparams["show progress"]
        relative_rbf_distance = hparams["metric"] == "relative rbf distance" # transform the distances nonlinearly with 1 - exp(- (Dhd - min(Dhd))/(2*std(Dhd)) ) as described in the paper
        n_iter                = hparams["n iter"]
        LR                    = hparams["LR"]
        # decay = np.exp(np.log(1e-3) / n_iter) # aim for an end LR of 1e-3 , if not initialised with a std of 10 (not recommended), then this value should be changed as well
        decay = np.exp(np.log(1e-4) / n_iter) # aim for an end LR of 1e-3 , if not initialised with a std of 10 (not recommended), then this value should be changed as well
        squared_D  = False
        stop_D_exa = 0
        if hparams["exaggerate D"]: # exaggeration of HD distances by taking them squared
            stop_D_exa = int(n_iter*hparams["stop exaggeration"]) # iteration when we stop the exaggeration
            squared_D  = True

        perms         = np.arange(N)
        batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
        grad_acc      = np.ones((N, 2))
        Dhd_quartet   = np.zeros((6,))
        momentums     = np.zeros((N, 2))
        for i in range(n_iter):
            LR *= decay
            if i == stop_D_exa:
                squared_D = False
            np.random.shuffle(perms)
            fast_distance_scaling_update_momentum(N, Xld, LR, perms, batches_idxes, grad_acc, Xhd, squared_D, Dhd_quartet, relative_rbf_distance)
            if show_progress and i % 50 == 0:
                progress_listener.notify((self.dataset_name, self.proj_name, Xld, self), [])

@numba.jit(nopython=True, fastmath = True)
def fast_distance_scaling_update_momentum(momentum, N, X_LD, LR, perms, batches_idxes, grad_acc,  Xhd, squared_D, Dhd_quartet, relative_rbf_distance):
    grad_acc.fill(0.)
    beta = 1. - momentum_alpha
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

        if relative_rbf_distance and i < int(0.25*n_iter):
            quartet_grads = compute_quartet_grads(LD_points, relative_rbf_dists(Dhd_quartet))
        else:
            Dhd_quartet  /= np.sum(Dhd_quartet)
            quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet)

        momentums[quartet[0], 0] = momentum_alpha*momentums[quartet[0], 0] + beta*quartet_grads[0]
        momentums[quartet[0], 1] = momentum_alpha*momentums[quartet[0], 1] + beta*quartet_grads[1]
        momentums[quartet[1], 0] = momentum_alpha*momentums[quartet[1], 0] + beta*quartet_grads[2]
        momentums[quartet[1], 1] = momentum_alpha*momentums[quartet[1], 1] + beta*quartet_grads[3]
        momentums[quartet[2], 0] = momentum_alpha*momentums[quartet[2], 0] + beta*quartet_grads[4]
        momentums[quartet[2], 1] = momentum_alpha*momentums[quartet[2], 1] + beta*quartet_grads[5]
        momentums[quartet[3], 0] = momentum_alpha*momentums[quartet[3], 0] + beta*quartet_grads[6]
        momentums[quartet[3], 1] = momentum_alpha*momentums[quartet[3], 1] + beta*quartet_grads[7]


        grad_acc[quartet[0]] += momentums[quartet[0]]
        grad_acc[quartet[1]] += momentums[quartet[1]]
        grad_acc[quartet[2]] += momentums[quartet[2]]
        grad_acc[quartet[3]] += momentums[quartet[3]]
    X_LD -= LR*grad_acc





@numba.jit(nopython=True, fastmath = True)
def fast_distance_scaling_update(N, X_LD, LR, perms, batches_idxes, grad_acc,  Xhd, squared_D, Dhd_quartet, relative_rbf_distance):
    grad_acc.fill(0.)

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

        if relative_rbf_distance:
            quartet_grads = compute_quartet_grads(LD_points, relative_rbf_dists(Dhd_quartet))
        else:
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

    X_LD -= LR*grad_acc

@numba.jit(nopython=True)
def relative_rbf_dists(Dhd_quartet):
    rel_dists = np.exp((Dhd_quartet-np.min(Dhd_quartet)) / (-2*np.std(Dhd_quartet)))
    rel_dists = 1 - rel_dists
    rel_dists /= np.sum(rel_dists)
    return rel_dists
