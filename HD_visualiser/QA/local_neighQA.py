from QA.local_QA import local_QA
import numpy as np
import numba

class local_neighQA(local_QA):
    def __init__(self, N):
        super(local_neighQA, self).__init__(N)
        self.top_right_title           = "top right local neighQA values"
        self.top_right_description     = ""
        self.top_right_scores_vs_rand  = None # will be np array of shape (N,) with values between 0 and 1
        self.top_right_scores_vs_self  = None # will be np array of shape (N,) with values between 0 and 1

        self.bottom_right_title        = "bottom right local neighQA values"
        self.bottom_right_description     = ""
        self.bottom_right_scores_vs_rand  = None # will be np array of shape (N,) with values between 0 and 1
        self.bottom_right_scores_vs_self  = None # will be np array of shape (N,) with values between 0 and 1

        self.bottom_left_title            = "bottom left local neighQA values"
        self.bottom_left_description      = ""
        self.bottom_left_scores_vs_rand   = None # will be np array of shape (N,) with values between 0 and 1
        self.bottom_left_scores_vs_self   = None # will be np array of shape (N,) with values between 0 and 1
        self.ready = False


    def do_local_QA(self, N, anchor_idxes, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
        top_right_title,    top_right_scores_vs_rand,    top_right_scores_vs_self,    top_right_overall_score,   top_right_description     = furthest_neighbour(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
        bottom_right_title, bottom_right_scores_vs_rand, bottom_right_scores_vs_self, bottom_right_overall_score, bottom_right_description = anchor_ranking_preservation(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
        bottom_left_title,  bottom_left_scores_vs_rand,  bottom_left_scores_vs_self,  bottom_left_overall_score, bottom_left_description   = local_manifold_tearing(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification)
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

@numba.jit(nopython=True, fastmath=True)
def anchor_ranking_preservation(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    title = "preservation of ranking with anchor points"
    description = "A measure of the preservation of the ranking of the anchor points. For each point,\
     the difference in ranking for each anchor point between HD and LD is transformed as a value between 0 and 1 by passing its \
     negative value divided by (N_anchors*0.1) in an exponential function. These are accumulated and then scaled between 0 and one :\
      (score_i - random_score_from_random_simulation) / (perfect_score - random_score_from_random_simulation). Looking at\
       the code in  anchor_ranking_preservation() from the file local_neighQA.py might be helpfull to understand this score."


    N_anchors = anchor_idxes.shape[0]
    gaussians_std = N_anchors/10
    gaussians = np.exp(-np.arange(N_anchors) / gaussians_std)
    coeffs    = np.ones((N_anchors))

    anchor_D_hd   = D_hd[:, anchor_idxes]
    anchor_D_ld   = D_ld[:, anchor_idxes]
    anchor_D_rand = D_ld_rand[:, anchor_idxes]

    rank_score = np.zeros((N,))
    E_rand = 0.
    for point in range(N):
        Dhd_anchors   = anchor_D_hd[point] # distance from this point to all anchors
        Dld_anchors   = anchor_D_ld[point]
        Drand_anchors = anchor_D_rand[point]

        sorted_hd_anchors   = Dhd_anchors.argsort(kind='mergesort') # gives the ranking for the anchors
        sorted_ld_anchors   = Dld_anchors.argsort(kind='mergesort')
        sorted_rand_anchors = Drand_anchors.argsort(kind='mergesort')

        point_score = 0.
        for i in range(N_anchors):
            hd_anchor_idx = sorted_hd_anchors[i]
            for ld_i in range(N_anchors):
                if sorted_ld_anchors[ld_i] == hd_anchor_idx:
                    point_score += gaussians[np.abs(ld_i - i)] * coeffs[i]
                if sorted_rand_anchors[ld_i] == hd_anchor_idx:
                    E_rand      += gaussians[np.abs(ld_i - i)] * coeffs[i]

        rank_score[point] = point_score
    E_rand /= N
    perfect = np.sum(coeffs)

    scores_vs_rand = np.maximum(rank_score - E_rand, 0.) / (perfect - E_rand)
    min_score, max_score = np.min(rank_score), np.max(rank_score)
    scores_vs_self = (rank_score - min_score) / (max_score - min_score)
    return title, scores_vs_rand, scores_vs_self,  np.mean(scores_vs_rand), description


@numba.jit(nopython=True, fastmath=True)
def furthest_neighbour(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    title = "preservation of furthest neighbours"
    description = "intersection of furthest neighbours:\n This score reflects the intersection of the sets of 20% furthest points in HD and in LD. 1 means perfect overlap, 0 is random.\n \
        The score is computed as follows : score_i = (intersection_i - Expected_random_intersection) / (perfect_intersection - Expected_random_intersection)"

    proportion = 0.2
    N_furthest = max(2, int(proportion * N))

    bool_HD = np.zeros((N, N), dtype=numba.boolean)
    bool_LD = np.zeros((N, N), dtype=numba.boolean)
    intersections = np.zeros((N,))

    for pt in range(N):
        for other in range(1, N_furthest+1):
            bool_HD[pt, neigh_hd[pt, -other]] = True
            bool_LD[pt, neigh_ld[pt, -other]] = True

        intersections[pt] = np.mean(bool_HD[pt] * bool_LD[pt])

    intersection = bool_HD * bool_LD
    E_rand       = proportion*proportion
    perfect      = proportion

    scores_vs_rand = np.maximum(intersections - E_rand, 0.) / (perfect - E_rand)

    min_intersection, max_intersection = np.min(intersections), np.max(intersections)
    scores_vs_self = (intersections - min_intersection) / (max_intersection - min_intersection)

    return title, scores_vs_rand, scores_vs_self,  np.mean(scores_vs_rand), description

@numba.jit(nopython=True, fastmath=True)
def local_manifold_tearing(N, D_hd, X_hd, neigh_hd, D_ld, X_ld, neigh_ld, anchor_idxes, D_ld_rand, X_ld_rand, neigh_ld_rand, is_labeled, is_classification):
    title = "local tearing of the manifold"
    description = "intersection of nearest neighbours:\n This score reflects the intersection of the sets of 3% nearest points in HD and in LD. 1 means perfect overlap, 0 is random.\n \
        The score is computed as follows : score_i = (intersection_i - Expected_random_intersection) / (perfect_intersection - Expected_random_intersection)"

    proportion = 0.03
    N_nearest  = max(2, int(proportion * N))

    bool_HD = np.zeros((N, N), dtype=numba.boolean)
    bool_LD = np.zeros((N, N), dtype=numba.boolean)
    intersections = np.zeros((N,))

    for pt in range(N):
        for other in range(N_nearest):
            bool_HD[pt, neigh_hd[pt, other]] = True
            bool_LD[pt, neigh_ld[pt, other]] = True

        intersections[pt] = np.mean(bool_HD[pt] * bool_LD[pt])

    intersection = bool_HD * bool_LD
    E_rand       = proportion*proportion
    perfect      = proportion

    scores_vs_rand = np.maximum(intersections - E_rand, 0.) / (perfect - E_rand)

    min_intersection, max_intersection = np.min(intersections), np.max(intersections)
    scores_vs_self = (intersections - min_intersection) / (max_intersection - min_intersection)

    return title, scores_vs_rand, scores_vs_self,  np.mean(scores_vs_rand), description
