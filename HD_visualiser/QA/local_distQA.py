from QA.local_QA import local_QA
import numpy as np
import numba
from scipy.stats import pearsonr

class local_distQA(local_QA):
    def __init__(self, N):
        super(local_distQA, self).__init__(N)
        self.top_right_title              = "top right local distQA values"
        self.top_right_description        = "" # will be np array of shape (N,) with values between 0 and 1
        self.top_right_scores_vs_rand     = None # will be np array of shape (N,) with values between 0 and 1
        self.top_right_scores_vs_self     = None # will be np array of shape (N,) with values between 0 and 1
        self.top_right_overall_score      = None # overall score

        self.bottom_right_title           = "bottom right local distQA values"
        self.bottom_right_description     = "" # will be np array of shape (N,) with values between 0 and 1
        self.bottom_right_scores_vs_rand  = None # will be np array of shape (N,) with values between 0 and 1
        self.bottom_right_scores_vs_self  = None # will be np array of shape (N,) with values between 0 and 1
        self.bottom_right_overall_score   = None # overall score
        self.ready = False


    def do_local_QA(self, N, anchor_idxes, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
        top_right_title,    top_right_scores_vs_rand,    top_right_scores_vs_self, top_right_overall_score,   top_right_description    = rel_anchor_dist(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
        bottom_right_title, bottom_right_scores_vs_rand, bottom_right_scores_vs_self, bottom_right_overall_score, bottom_right_description = rel_local_dist(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
        self.top_right_title             = top_right_title
        self.top_right_description       = top_right_description
        self.top_right_scores_vs_rand    = top_right_scores_vs_rand
        self.top_right_scores_vs_self    = top_right_scores_vs_self
        self.top_right_overall_score     = np.round(top_right_overall_score, 3)
        self.bottom_right_title          = bottom_right_title

        self.bottom_right_description    = bottom_right_description
        self.bottom_right_scores_vs_rand = bottom_right_scores_vs_rand
        self.bottom_right_scores_vs_self = bottom_right_scores_vs_self
        self.bottom_right_overall_score  = np.round(bottom_right_overall_score, 3)
        self.ready = True


@numba.jit(nopython=True, fastmath=True)
def rel_anchor_dist(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    title = "correlation: distances to anchors"
    description = "Distance correlation towards anchor points :\n\
     Anchor points are determined when a new dataset is opened (therefore all projections use the same anchors)\n\
     Anchor points are the centroids of recursive Kmeans in the HD dataset. These centroids are real points, \
            and the idea behind recursive clustering is to have a good representation of the global structure of the data.\n \
     The score is simply the correlation between HD and LD for the distances for each point towards the anchor points. \n \
     "

    Dhd_to_anchors   = D_hd[:, anchor_idxes]
    Dld_to_anchors   = D_ld[:, anchor_idxes]

    corr      = np.zeros((N,))
    for i in range(N):
        centered_hd = Dhd_to_anchors[i] - np.mean(Dhd_to_anchors[i])
        centered_ld = Dld_to_anchors[i] - np.mean(Dld_to_anchors[i])
        corr[i] = np.sum(centered_hd * centered_ld)/(1e-8+np.sqrt(np.sum(centered_hd**2) * np.sum(centered_ld**2)))

    min_corr, max_corr = np.min(corr), np.max(corr)
    corr_v_self = (corr - min_corr) / (1e-8 + max_corr - min_corr)
    return title, corr, corr_v_self, np.mean(corr), description


@numba.jit(nopython=True, fastmath=True)
def rel_local_dist(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    title = "correlation: distances to nearby points"
    description = "Distance correlation towards the 5% nearest neighbours in HD (which aren't necessarily the nearset neighbours in LD) \n"

    N_neighbours    = max(5, int(0.05*N))
    neighbour_idxes = neigh_hd[:, :N_neighbours]

    corr      = np.zeros((N,))
    for i in range(N):
        Dhd_to_neighbours = D_hd[i][neighbour_idxes[i]]
        Dld_to_neighbours = D_ld[i][neighbour_idxes[i]]

        centered_hd = Dhd_to_neighbours - np.mean(Dhd_to_neighbours)
        centered_ld = Dld_to_neighbours - np.mean(Dld_to_neighbours)
        corr[i] = np.sum(centered_hd * centered_ld)/(1e-8+np.sqrt(np.sum(centered_hd**2) * np.sum(centered_ld**2)))


    min_corr, max_corr = np.min(corr), np.max(corr)
    corr_v_self = (corr - min_corr) / (1e-8+ max_corr - min_corr)
    return title, corr, corr_v_self, np.mean(corr), description
