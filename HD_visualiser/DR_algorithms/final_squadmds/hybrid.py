import numpy as np
import numba
# from tSNE import joint_P, KL_divergeance_BH
# from SQuaD_MDS import fill_MDS_grads, init_embedding
from DR_algorithms.final_squadmds.tSNE import joint_P, KL_divergeance_BH
from DR_algorithms.final_squadmds.SQuaD_MDS import fill_MDS_grads, init_embedding, nestrov_iteration
from sklearn.manifold._t_sne import _kl_divergence_bh

from sklearn.metrics.pairwise import cosine_similarity



import time
from scipy.stats import zscore



def populate_hparams(hparams, N):
    if not 'barnes-hut' in hparams:
        hparams['barnes-hut'] = True

    if not 'on N PC' in hparams:
        hparams['on N PC'] = -1

    if not 'n iter' in hparams:
        hparams['n iter'] = 1000 # higher values bring better results

    if not 'tSNE LR' in hparams:
        if hparams['barnes-hut']:
            hparams['tSNE LR'] = 0.005 * N
        else:
            hparams['tSNE LR'] = 1

    if not 'MDS LR' in hparams:
        if hparams['barnes-hut']:
            hparams['MDS LR'] = hparams['tSNE LR'] / 6
        else:
            # hparams['MDS LR'] = 0.01
            hparams['MDS LR'] = 0.01

    if not 'tSNE perplexities' in hparams:
        hparams['tSNE perplexities'] = [4., 50.]

    if not 'tSNE exa' in hparams:
        hparams['tSNE exa'] = 2.



def nestrov_iteration_FItSNE( Xld, mds_grads, tsne_grads, momentums, momentums2, perms, batches_idxes, Xhd, Dhd_quartet, LR_MDS, LR_tSNE, _P_, N): # coputes and applies gradients, updates momentum too
    mul_MDS   = LR_MDS
    mul_tSNE  = LR_tSNE
    momentums   *= 0.99
    momentums2  *= 0.99

    temp_params = Xld + momentums + momentums2

    mds_grads.fill(0.)
    tsne_grads.fill(0.)

    fill_MDS_grads(temp_params, mds_grads, perms, batches_idxes, Xhd, Dhd_quartet, exaggeration=False)
    DOF = 1.
    tsne_grads = KL_divergeance_BH(temp_params.ravel(), _P_, DOF, N, 2, 0, False, grad=tsne_grads)

    std_norm_MDS  = np.std(np.linalg.norm(mds_grads, axis=1, keepdims=True))
    std_norm_tSNE = np.std(np.linalg.norm(tsne_grads, axis=1, keepdims=True))
    mul_MDS  = LR_MDS  / std_norm_MDS
    mul_tSNE = LR_tSNE / std_norm_tSNE

    momentums  -=  (mul_MDS * mds_grads)
    momentums2 -=  (mul_tSNE * tsne_grads)

    Xld += momentums + momentums2

def nestrov_iteration_BH(Xld, mds_grads, tsne_grads, momentums, perms, batches_idxes, Xhd, Dhd_quartet, LR_MDS, LR_tSNE, _P_, N): # coputes and applies gradients, updates momentum too
    momentums  *= 0.995
    temp_params = Xld + momentums

    mds_grads.fill(0.)
    tsne_grads.fill(0.)

    fill_MDS_grads(temp_params, mds_grads, perms, batches_idxes, Xhd, Dhd_quartet, exaggeration=False)
    DOF = 1.
    tsne_grads = KL_divergeance_BH(temp_params.ravel(), _P_, DOF, N, 2, 0, False, grad=tsne_grads)

    norm_MDS  = np.linalg.norm(mds_grads, keepdims=True)
    norm_tSNE = np.linalg.norm(tsne_grads, keepdims=True)
    mul_MDS = 0.; mul_tSNE = 0.
    if norm_MDS > 1e-12:
        mul_MDS  = LR_MDS / norm_MDS

    if norm_tSNE > 1e-12:
        mul_tSNE = LR_tSNE / norm_tSNE

    momentums -= ( mul_MDS * mds_grads +  mul_tSNE * tsne_grads)
    Xld += momentums


def run_hybrid(Xhd, hparams, progress_stuff=None):
    N, M = Xhd.shape
    populate_hparams(hparams, N) # set the missing hyperparameters with their default values

    target_scale = 25.
    Xld = init_embedding(Xhd, target = target_scale)

    momentums_MDS   = np.zeros((N, 2))
    momentums_tSNE  = np.zeros((N, 2))
    tsne_grads      = np.zeros((N, 2), dtype=np.float32)
    mds_grads       = np.zeros((N, 2), dtype=np.float32)
    perms           = np.arange(N)
    batches_idxes   = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
    Dhd_quartet     = np.zeros((6,))

    N_iter = hparams['n iter']
    LR_tSNE_init = hparams['tSNE LR']
    LR_tSNE      = LR_tSNE_init
    LR_MDS_init  = hparams['MDS LR']
    LR_MDS       = LR_MDS_init
    decay_cte    = 0.28
    decay_offset = -np.exp(-1/decay_cte)
    BH_version   = hparams['barnes-hut']
    force_scale  = not BH_version

    perplexities_list = hparams['tSNE perplexities']
    # perplexities_list = [2, 4, 8, 16, 32, 64]
    _P_ = joint_P(Xhd, perplexities_list[0])
    if len(perplexities_list) > 1:
        for PP in perplexities_list[1:]:
            _P_ += joint_P(Xhd, PP)
        _P_ /= len(perplexities_list)
    _P_ *= hparams['tSNE exa']
    tsne_exa_stop = min(500, int(0.2*N_iter))
    decay_start = tsne_exa_stop

    progress_listener, instance = progress_stuff
    for i in range(N_iter):
        if i == tsne_exa_stop:  # stop tSNE early exaggeration
            _P_ /= hparams['tSNE exa']
        if i > decay_start:
            ratio = (i - decay_start) / (N_iter - decay_start)
            mul = (np.exp(-(ratio*ratio) / decay_cte) + decay_offset)
            LR_tSNE = LR_tSNE_init * mul
            LR_MDS  = LR_MDS_init  * mul


        np.random.shuffle(perms) # used for the random quartet designation

        if BH_version:
            nestrov_iteration_BH(Xld, mds_grads, tsne_grads, momentums_MDS, perms, batches_idxes, Xhd, Dhd_quartet, LR_MDS, LR_tSNE, _P_, N)
        else:
            if i == max(0, N_iter - 150):
                force_scale = False
            if force_scale:
                # stdev = np.std(Xld, axis = 0)
                stdev = np.std(Xld)
                transformation = target_scale / stdev
                Xld *= transformation
                momentums_MDS  *= transformation
                momentums_tSNE *= transformation
            nestrov_iteration_FItSNE(Xld, mds_grads, tsne_grads, momentums_MDS, momentums_tSNE, perms, batches_idxes, Xhd, Dhd_quartet, LR_MDS, LR_tSNE, _P_, N)
            print(np.std(Xld, axis = 0))
        if i % 10 == 0:
            progress_listener.notify((instance.dataset_name, instance.proj_name, Xld, instance), [])

    return Xld
