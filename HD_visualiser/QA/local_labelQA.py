from QA.local_QA import local_QA
import numba
import numpy as np
from QA.general_QA import  eval_auc



class local_labelQA(local_QA):
    def __init__(self, N):
        super(local_labelQA, self).__init__(N)
        self.top_right_title           = "top right local labelQA values"
        self.top_right_description     = ""
        self.top_right_scores_vs_rand  = None # will be np array of shape (N,) with values between 0 and 1
        self.top_right_scores_vs_self  = None # will be np array of shape (N,) with values between 0 and 1
        self.top_right_overall_score   = None # overall score

        self.bottom_right_title        = "bottom right local labelQA values"
        self.bottom_right_description     = ""
        self.bottom_right_scores_vs_rand  = None # will be np array of shape (N,) with values between 0 and 1
        self.bottom_right_scores_vs_self  = None # will be np array of shape (N,) with values between 0 and 1
        self.bottom_right_overall_score   = None # overall score

        self.bottom_left_title            = "bottom left local labelQA values"
        self.bottom_left_description      = ""
        self.bottom_left_scores_vs_rand   = None # will be np array of shape (N,) with values between 0 and 1
        self.bottom_left_scores_vs_self   = None # will be np array of shape (N,) with values between 0 and 1
        self.bottom_left_overall_score    = None # overall score

        self.ready = False


    def do_local_QA(self, N, anchor_idxes,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
        if not is_labeled:
            return
        top_right_title,    top_right_scores_vs_rand,    top_right_scores_vs_self, top_right_overall_score,    top_right_description       = MS_uncertainty_preservation(N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
        bottom_right_title, bottom_right_scores_vs_rand, bottom_right_scores_vs_self, bottom_right_overall_score, bottom_right_description = MS_prediction_agreement(N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
        bottom_left_title,  bottom_left_scores_vs_rand,  bottom_left_scores_vs_self, bottom_left_overall_score,  bottom_left_description   = small_scale_prediction_agreement(N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
        self.top_right_title               = top_right_title
        self.top_right_description         = top_right_description
        self.top_right_scores_vs_rand      = top_right_scores_vs_rand
        self.top_right_scores_vs_self      = top_right_scores_vs_self
        self.top_right_overall_score       = np.round(top_right_overall_score, 3)

        self.bottom_right_title            = bottom_right_title
        self.bottom_right_description      = bottom_right_description
        self.bottom_right_scores_vs_rand   = bottom_right_scores_vs_rand
        self.bottom_right_scores_vs_self   = bottom_right_scores_vs_self
        self.bottom_right_overall_score    = np.round(bottom_right_overall_score, 3)

        self.bottom_left_title             = bottom_left_title
        self.bottom_left_description       = bottom_left_description
        self.bottom_left_scores_vs_rand    = bottom_left_scores_vs_rand
        self.bottom_left_scores_vs_self    = bottom_left_scores_vs_self
        self.bottom_left_overall_score     = np.round(bottom_left_overall_score, 3)
        self.ready = True



def MS_uncertainty_preservation(N, Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    title = "label uncertainty conservation"
    description = "   "

    scales  = [3]
    weights = [1.]
    while scales[-1]*2 < int(N*0.1):
        new_scale = scales[-1]*2
        if new_scale % 2 == 0:
            new_scale += 1
        scales.append(new_scale)
        weights.append(1.)
    scales = np.array(scales)
    weights = np.array(weights)



    if is_classification:
        scores_vs_rand, scores_vs_self, overall_score = classification_MS_uncertainty_preservation(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
    else:
        scores_vs_rand, scores_vs_self, overall_score = regression_MS_uncertainty_preservation(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)

    return title, scores_vs_rand, scores_vs_self, overall_score, description



def MS_prediction_agreement(N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    title = "multi-scale label agreement"
    description = " "

    scales  = [3]
    weights = [1.]
    while scales[-1]*2 < int(N*0.5):
        new_scale = scales[-1]*2
        if new_scale % 2 == 0:
            new_scale += 1
        scales.append(new_scale)
        weights.append(1.)
    scales = np.array(scales)
    weights = np.array(weights)


    if is_classification:
        scores_vs_rand, scores_vs_self, overall_score = classification_MS_label_agreement(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
    else:
        scores_vs_rand, scores_vs_self, overall_score = regression_MS_label_agreement(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)

    return title, scores_vs_rand, scores_vs_self, overall_score, description

def small_scale_prediction_agreement(N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    title = "small scale label agreement"
    description = " "

    scales = np.array([1, 3])
    weights = np.array([1, 0.5])

    if is_classification:
        scores_vs_rand, scores_vs_self, overall_score = classification_MS_label_agreement(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
    else:
        scores_vs_rand, scores_vs_self, overall_score = regression_MS_label_agreement(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)

    return title, scores_vs_rand, scores_vs_self, overall_score, description







@numba.jit(nopython=True, fastmath=True)
def classification_MS_label_agreement(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    N_classes = np.unique(Y).shape[0]
    N_scales  = scales.shape[0]
    N_classes_times_2 = 2*N_scales
    sum_weights = np.sum(weights)
    smoothing = (1e-7)/N_classes

    P_HD, P_LD, P_rand = np.zeros((N_classes,)), np.zeros((N_classes,)), np.zeros((N_classes,))
    MS_simi        = np.zeros((N,))
    simi          = np.zeros((N, N_scales))
    simi_rand_acc = np.zeros((N_scales,))

    for scale in range(N_scales):
        K = scales[scale]
        for point in range(N):
            P_HD.fill(smoothing); P_LD.fill(smoothing); P_rand.fill(smoothing)
            knn_HD, knn_LD, knn_rand = neigh_hd[point][:K], neigh_ld[point][:K], neigh_ld_rand[point][:K]
            for other in range(K):
                label_HD, label_LD, label_rand = Y[knn_HD[other]], Y[knn_LD[other]], Y[knn_rand[other]]
                P_HD[label_HD]     += 1
                P_LD[label_LD]     += 1
                P_rand[label_rand] += 1
            P_HD /= K; P_LD /= K; P_rand /= K

            M_HDLD   = 0.5*(P_HD + P_LD)
            M_HDrand = 0.5*(P_HD + P_rand)

            KL_LD, KL_HDLD = 0., 0.
            KL_rand, KL_HDrand = 0., 0.
            for p in range(N_classes):
                KL_LD   += P_LD[p]*np.log(P_LD[p]/M_HDLD[p])
                KL_HDLD += P_HD[p]*np.log(P_HD[p]/M_HDLD[p])

                KL_rand   += P_rand[p]*np.log(P_rand[p]/M_HDrand[p])
                KL_HDrand += P_HD[p]*np.log(P_HD[p]/M_HDrand[p])
            div_LD = (KL_LD + KL_HDLD) / 2
            div_rand = (KL_rand + KL_HDrand) / 2

            simi[point, scale] = 1 - div_LD
            simi_rand_acc[scale] += (1 - div_rand)
        simi_rand_acc[scale] /= N

    MS_simi_rand = np.sum(simi_rand_acc*weights) / sum_weights
    for point in range(N):
        MS_simi[point] = np.sum(simi[point] * weights) / sum_weights

    scores_vs_rand = np.maximum((MS_simi - MS_simi_rand),0.)/(1 - MS_simi_rand)
    min_simi = np.min(MS_simi)
    scores_vs_self = (MS_simi - min_simi) / (np.max(MS_simi) - min_simi)
    return scores_vs_rand, scores_vs_self, np.mean(scores_vs_rand)

@numba.jit(nopython=True, fastmath=True)
def regression_MS_label_agreement(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    N_scales  = scales.shape[0]
    sum_weights = np.sum(weights)

    MS_disagreement = np.zeros((N,))
    disagreement          = np.zeros((N, N_scales))
    disagreement_rand_acc = np.zeros((N_scales,))

    for scale in range(N_scales):
        K = scales[scale]
        for point in range(N):
            knn_HD, knn_LD, knn_rand = neigh_hd[point][:K], neigh_ld[point][:K], neigh_ld_rand[point][:K]
            pred_HD, pred_LD, pred_rand = 0., 0., 0.
            for other in range(K):
                pred_HD   += Y[knn_HD[other]]
                pred_LD   += Y[knn_LD[other]]
                pred_rand += Y[knn_rand[other]]
            pred_HD   /= K
            pred_LD   /= K
            pred_rand /= K

            disagreement[point, scale]    = np.abs(pred_LD   - pred_HD)
            disagreement_rand_acc[scale] += np.abs(pred_rand - pred_HD)
        disagreement_rand_acc[scale] /= N

    MS_disagreement_rand = np.sum(disagreement_rand_acc*weights) / sum_weights
    for point in range(N):
        MS_disagreement[point] = np.sum(disagreement[point] * weights) / sum_weights

    scores_vs_rand = np.maximum(MS_disagreement_rand - MS_disagreement, 0.) / MS_disagreement_rand
    min_simi = np.min(MS_disagreement)
    scores_vs_self = 1 - (MS_disagreement - min_simi + 1e-8) / (np.max(MS_disagreement) - min_simi + 1e-8)
    return scores_vs_rand, scores_vs_self, np.mean(scores_vs_rand)




























@numba.jit(nopython=True, fastmath=True)
def classification_MS_uncertainty_preservation(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    N_classes = np.unique(Y).shape[0]
    N_scales  = scales.shape[0]
    sum_weights = np.sum(weights)

    P_HD, P_LD, P_rand = np.zeros((N_classes,)), np.zeros((N_classes,)), np.zeros((N_classes,))
    MS_simi        = np.zeros((N,))
    Hsimi          = np.zeros((N, N_scales))
    Hsimi_rand_acc = np.zeros((N_scales,))

    for scale in range(N_scales):
        K = scales[scale]
        for point in range(N):
            P_HD.fill(0); P_LD.fill(0); P_rand.fill(0)
            knn_HD, knn_LD, knn_rand = neigh_hd[point][:K], neigh_ld[point][:K], neigh_ld_rand[point][:K]
            for other in range(K):
                label_HD, label_LD, label_rand = Y[knn_HD[other]], Y[knn_LD[other]], Y[knn_rand[other]]
                P_HD[label_HD]     += 1
                P_LD[label_LD]     += 1
                P_rand[label_rand] += 1
            P_HD /= K; P_LD /= K; P_rand /= K

            H_HD, H_LD, H_rand = 0., 0., 0.
            for p in range(N_classes):
                if P_HD[p] != 0.:
                    H_HD   -= P_HD[p]   * np.log(P_HD[p])
                if P_LD[p] != 0.:
                    H_LD   -= P_LD[p]   * np.log(P_LD[p])
                if P_rand[p] != 0.:
                    H_rand -= P_rand[p] * np.log(P_rand[p])

            Hsimi[point, scale]   = ( 1e-8 + 2*H_LD*H_HD) / ( 1e-8 + H_LD**2 + H_HD**2)
            Hsimi_rand_acc[scale] += ( 1e-8 + 2*H_rand*H_HD) / ( 1e-8 + H_rand**2 + H_HD**2)
        Hsimi_rand_acc[scale] /= N

    MS_simi_rand = np.sum(Hsimi_rand_acc*weights) / sum_weights
    for point in range(N):
        MS_simi[point] = np.sum(Hsimi[point] * weights) / sum_weights

    scores_vs_rand = np.maximum((MS_simi - MS_simi_rand),0.)/(1 - MS_simi_rand)
    min_simi = np.min(MS_simi)
    scores_vs_self = (MS_simi - min_simi) / (np.max(MS_simi) - min_simi)
    return scores_vs_rand, scores_vs_self, np.mean(scores_vs_rand)

@numba.jit(nopython=True, fastmath=True)
def regression_MS_uncertainty_preservation(scales, weights, N,  Y, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    N_scales  = scales.shape[0]
    sum_weights = np.sum(weights)

    MS_simi       = np.zeros((N,))
    simi          = np.zeros((N, N_scales))
    simi_rand_acc = np.zeros((N_scales,))

    for scale in range(N_scales):
        K = scales[scale]
        for point in range(N):
            knn_HD, knn_LD, knn_rand = neigh_hd[point][:K], neigh_ld[point][:K], neigh_ld_rand[point][:K]

            std_HD   = np.std(Y[knn_HD])
            std_LD   = np.std(Y[knn_LD])
            std_rand = np.std(Y[knn_rand])

            simi[point, scale]    = ( 1e-8 + 2*std_LD*std_HD) / ( 1e-8 + std_LD**2 + std_HD**2)
            simi_rand_acc[scale] += ( 1e-8 + 2*std_rand*std_HD) / ( 1e-8 + std_rand**2 + std_HD**2)
        simi_rand_acc[scale] /= N

    MS_simi_rand = np.sum(simi_rand_acc*weights) / sum_weights
    for point in range(N):
        MS_simi[point] = np.sum(simi[point] * weights) / sum_weights

    scores_vs_rand = np.maximum((MS_simi - MS_simi_rand),0.)/(1 - MS_simi_rand)
    min_simi = np.min(MS_simi)
    scores_vs_self = (MS_simi - min_simi + 1e-8) / (np.max(MS_simi) - min_simi + 1e-8)
    return scores_vs_rand, scores_vs_self, np.mean(scores_vs_rand)
