from QA.local_neighQA import local_neighQA
from QA.local_distQA  import local_distQA
from QA.local_labelQA import local_labelQA
from QA.general_QA import eval_dr_quality, generalQA, knngain, knngain_regression
import time
from engine.gui.event_ids import *
from engine.gui.listener import Listener
import threading
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree
from scipy.stats import pearsonr

def should_abort(model, embedding_instance, dataset_instance):
    return model.deleted or embedding_instance.deleted  or dataset_instance.deleted

def find_neighbours(X):
    tree       = KDTree(X)
    neighbours = tree.query(X, X.shape[0], return_distance=False)[:, 1:]
    return tree, neighbours

def project_then_QA(embedding_ID, proj_done_listener, error_listener, KNN_done_listener, progress_listener, general_Rnx_listener, Dcorr_listener, KNNgain_listener, local_distQA_listener, local_neighQA_listener, local_labelQA_listener, model, dataset_instance, embedding_instance):
    '''
    this function is executed  by a thread different from the main thread
    it does the projection and then computes the associated QA
    these are all done within try/catch statements because things can get deleted by the main thread. Using less locks means a smoother experience for the user
    '''
    # ~~~~~~~~~~~~~~~~~~~~~   compute embedding   ~~~~~~~~~~~~~~~~~~~~~
    X_LD = model.fit_transform(progress_listener, dataset_instance.X, dataset_instance.Y, dataset_instance.is_dists)
    if should_abort(model, embedding_instance, dataset_instance):
        return
    proj_done_listener.notify((embedding_ID, X_LD), [])
    # try:
    #     X_LD = model.fit_transform(progress_listener, dataset_instance.X, dataset_instance.Y)
    #     if should_abort(model, embedding_instance, dataset_instance):
    #         return
    #     proj_done_listener.notify((embedding_ID, X_LD), [])
    # except Exception as e:
    #     print("\n problem during DR for ", embedding_ID, " \n ", e, " \n\n")
    #     error_listener.notify(embedding_ID, [])
    #     return




    # ~~~~~~~~~~~~~~~~~~~~~   wait for Dataset to initialise    ~~~~~~~~~~~~~~~~~~~~~
    while not dataset_instance.initialised:
        time.sleep(0.7)
        if should_abort(model, embedding_instance, dataset_instance):
            return



    # ~~~~~~~~~~~~~~~~~~~~~    Rnx(K) curve    ~~~~~~~~~~~~~~~~~~~~~
    if should_abort(model, embedding_instance, dataset_instance):
        return
    try:
        D_LD     = pairwise_distances(X_LD)
        rnx, auc = eval_dr_quality(d_hd=dataset_instance.D, d_ld=D_LD)
        general_Rnx_listener.notify((embedding_ID, rnx, auc), [])
    except Exception as e:
        print("\n problem during computation of Rnx curve for ", embedding_ID, " \n ", e, " \n\n")
        error_listener.notify(embedding_ID, [])
        return


    # ~~~~~~~~~~~~~~~~~~~~~    embedding variables used in local QA and in point selection    ~~~~~~~~~~~~~~~~~~~~~
    if should_abort(model, embedding_instance, dataset_instance):
        return
    try:
        tree, neighbours = find_neighbours(X_LD)
        KNN_done_listener.notify((embedding_ID, tree, neighbours, D_LD), [])
    except Exception as e:
        print("\n problem during computation of Rnx curve for ", embedding_ID, " \n ", e, " \n\n")
        error_listener.notify(embedding_ID, [])
        return

    # ~~~~~~~~~~~~~~~~~~~~~    Dcorr    ~~~~~~~~~~~~~~~~~~~~~
    if should_abort(model, embedding_instance, dataset_instance):
        return
    try:
        Dcorr = np.round(pearsonr(D_LD.ravel(), dataset_instance.D.ravel())[0], 3)
        Dcorr_listener.notify((embedding_ID, Dcorr), [])
        is_labeled, is_classification = dataset_instance.is_labeled, dataset_instance.is_classification
    except Exception as e:
        print("\n problem during computation of distance correlation for: ", embedding_ID, " \n ", e, " \n\n")
        error_listener.notify(embedding_ID, [])
        return


    # ~~~~~~~~~~~~~~~~~~~~~   KNNgain curve    ~~~~~~~~~~~~~~~~~~~~~
    if should_abort(model, embedding_instance, dataset_instance):
        return
    try:
        if dataset_instance.is_labeled:
            if dataset_instance.is_classification:
                rnx, auc = knngain(d_hd=dataset_instance.D, d_ld=D_LD, labels = dataset_instance.Y)
            else:
                rnx, auc = knngain_regression(d_hd=dataset_instance.D, d_ld=D_LD, labels = dataset_instance.Y)
            KNNgain_listener.notify((embedding_ID, rnx, auc), [])
    except Exception as e:
        print("\n problem during computation of KNNgain curve: ", embedding_ID, " \n ", e, " \n\n")
        error_listener.notify(embedding_ID, [])
        return


    # ~~~~~~~~~~~~~~~~~~~~~   local dist QA    ~~~~~~~~~~~~~~~~~~~~~
    if should_abort(model, embedding_instance, dataset_instance):
        return
    try:
        embedding_instance.local_distQA.do_local_QA(X_LD.shape[0], dataset_instance.anchors,dataset_instance.D, dataset_instance.X, dataset_instance.neighbours,D_LD, X_LD, neighbours,dataset_instance.rand_D, dataset_instance.rand_X_ld, dataset_instance.rand_neighbours, is_labeled, is_classification)
        local_distQA_listener.notify(embedding_ID, [])
    except Exception as e:
        print("\n problem during local dist QA for ", embedding_ID, " \n ", e, " \n\n")
        error_listener.notify(embedding_ID, [])
        return



    # ~~~~~~~~~~~~~~~~~~~~~   local neighbour QA    ~~~~~~~~~~~~~~~~~~~~~
    if should_abort(model, embedding_instance, dataset_instance):
        return
    try:
        embedding_instance.local_neighQA.do_local_QA(X_LD.shape[0], dataset_instance.anchors, dataset_instance.D, dataset_instance.X, dataset_instance.neighbours,D_LD, X_LD, neighbours,dataset_instance.rand_D, dataset_instance.rand_X_ld, dataset_instance.rand_neighbours, is_labeled, is_classification)
        local_neighQA_listener.notify(embedding_ID, [])
    except Exception as e:
        print("\n problem during local neighbour QA for ", embedding_ID, " \n ", e, " \n\n")
        error_listener.notify(embedding_ID, [])
        return




    # ~~~~~~~~~~~~~~~~~~~~~   local label QA    ~~~~~~~~~~~~~~~~~~~~~
    if should_abort(model, embedding_instance, dataset_instance):
        return
    try:
        if dataset_instance.is_labeled:
            embedding_instance.local_labelQA.do_local_QA(X_LD.shape[0], dataset_instance.anchors, dataset_instance.Y, dataset_instance.D, dataset_instance.X, dataset_instance.neighbours,D_LD, X_LD, neighbours,dataset_instance.rand_D, dataset_instance.rand_X_ld, dataset_instance.rand_neighbours, is_labeled, is_classification)
            local_labelQA_listener.notify(embedding_ID, [])
    except Exception as e:
        print("\n problem during local label QA for ", embedding_ID, " \n ", e, " \n\n")
        error_listener.notify(embedding_ID, [])
        return




class Embedding():
    def __init__(self, dataset_name, projection_name, N, is_dists):
        self.deleted = False
        self.name = dataset_name + " ~ " + projection_name
        self.is_dists = is_dists
        self.dataset_name = dataset_name
        self.proj_name    = projection_name
        self.local_neighQA   = local_neighQA(N)
        self.local_distQA    = local_distQA(N)
        self.local_labelQA   = local_labelQA(N)
        self.generalQA       = generalQA(N)
        self.lock = threading.Lock()
        self.done = False
        self.X_LD  = None
        self.KD_tree      = None
        self.neighbours   = None
        self.D            = None
        self.neighbours_ready = False
        self.Shepard_distances = None

    def find_K_closest_to_cursor(self, LD_pos, K):
        if not self.neighbours_ready:
            return None
        center_idx = self.KD_tree.query(np.array(LD_pos), 1, return_distance=False)[0]
        return self.neighbours[center_idx, :K].ravel()

    def save(self, filename, dataset):
        if self.done and dataset is not None and not dataset.deleted:
            filename = filename.replace(" :: ", "_subset_")
            filename = filename.replace("+", "_with_")
            if dataset.is_labeled:
                XY = np.hstack((self.X_LD, dataset.Y.reshape((-1,1))))
            else:
                XY = np.hstack((self.X_LD, np.ones((self.X_LD.shape[0],1))))

            np.save(filename, XY)
            print("\nsaved embedding : "+self.name+"\nin  file : "+filename+"    (shape = (N, 3)  <-- first 2 dimensions = embedding, last dimension = label)\n")

    def start(self, dataset, model, manager):
        error_listener         = Listener(PROJECTION_ERROR,   [manager])
        proj_done_listener     = Listener(PROJECTION_DONE,   [manager])
        KNN_done_listener      = Listener(EMBEDDING_KNN_DONE,   [manager])
        progress_listener      = Listener(CONVERGENCE_UPDATE, [manager])
        general_Rnx_listener   = Listener(GENERAL_RNX, [manager])
        Dcorr_listener         = Listener(DCORR_DONE, [manager])
        KNNgain_listener       = Listener(KNN_GAIN_DONE, [manager])
        local_distQA_listener  = Listener(LOCAL_DISTQA, [manager])
        local_neighQA_listener = Listener(LOCAL_NEIGHQA, [manager])
        local_labelQA_listener = Listener(LOCAL_LABELQA, [manager])
        projection_thread = threading.Thread(target=project_then_QA, args=[self.name, proj_done_listener, error_listener, KNN_done_listener, progress_listener, general_Rnx_listener, Dcorr_listener, KNNgain_listener, local_distQA_listener, local_neighQA_listener, local_labelQA_listener, model, dataset, self])
        projection_thread.start()


    def delete(self):
        with self.lock:
            self.deleted = True
            if self.local_neighQA is not None:
                self.local_neighQA.delete()
                self.local_neighQA = None
            if self.local_distQA is not None:
                self.local_distQA.delete()
                self.local_distQA = None
            if self.local_labelQA is not None:
                self.local_labelQA.delete()
                self.local_labelQA = None
