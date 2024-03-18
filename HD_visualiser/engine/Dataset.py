import numpy as np
import threading
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import KDTree

def init_dataset(X, Dataset_instance):
    
    N, M = X.shape
    if not Dataset_instance.is_dists:
        D = pdist(X)
        n_dist = D.shape[0]
        D = squareform(D)
    else:
        D = X
    neighbours = D.argsort(axis=-1, kind='mergesort')


    X_PC = X.copy()
    if M > 12:
        X_PC = PCA(n_components=10, whiten=True, copy=True).fit_transform(X)
    X_PC += np.random.normal(size = (N*X_PC.shape[1])).reshape((N,X_PC.shape[1]))*0.000001*(np.std(X_PC, axis=0)+1e-6) # prevents obserations from beeing equal, which could be problematic during the following Kmeans

    centers = None

    if X.shape[0] < 50: # all the points are considered landmarks
        centers = np.arange(X.shape[0])
    else:
        if not Dataset_instance.is_dists:
            N_stop = max(10, int(0.04*X.shape[0]))
            centers    = []
            partitions = [np.arange(X_PC.shape[0])]
            while len(partitions) > 0:
                part = partitions.pop()
                if part.shape[0] >= N_stop:
                    n_clusters = 2
                else:
                    center_rel_idx = np.argmin(np.sum(D[part][:, part], axis=1))
                    centers.append(part[center_rel_idx])
                    continue

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_PC[part])

                # if part.shape[0] < 200:
                #     kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_PC[part])
                # else:
                #     kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X_PC[part])
                labels = kmeans.labels_
                for i in range(n_clusters):
                    new_part = part[np.where(labels == i)[0]]
                    if new_part.shape[0] > 0:
                        partitions.append(new_part)
        else:
            centers = np.random.choice(np.arange(X.shape[0]), size=49, replace=False)

    if len(centers) < 5: # should not be happening, but just in case
        print("couldn't find enough anchor points. Going for random points")
        centers = np.random.choice(np.arange(X.shape[0]), replace = False, size = min(30, X.shape[0]-1))

    rand_D = squareform(pdist(Dataset_instance.rand_X_ld))
    rand_neighbours = rand_D.argsort(axis=-1, kind='mergesort')
    if Dataset_instance is not None and not Dataset_instance.deleted:
        Dataset_instance.neighbours  = neighbours[:, 1:]
        Dataset_instance.anchors     = np.array(centers)
        Dataset_instance.D           = D
        Dataset_instance.rand_neighbours     = rand_neighbours[:, 1:]
        Dataset_instance.rand_D              = rand_D
        Dataset_instance.initialised = True




class Dataset():
    def __init__(self, name, X, Y, is_labeled, is_classification, is_dists):
        self.name      = name
        self.deleted   = False

        self.is_dists = is_dists
        N, M = X.shape
        self.X          = X
        self.D   = None
        self.X_mins     = np.min(X, axis=0)
        self.X_maxes    = np.max(X, axis=0)
        self.Y          = Y
        self.is_labeled = is_labeled
        self.is_classification = is_classification
        if is_dists:
            self.X_mins     = 0.
            self.X_maxes    = 0.
            self.D   = X
        # KNN_score(X, Y, is_classification)

        self.neighbours          = None
        self.rand_X_ld           = np.random.uniform(size = 2*N).reshape((N, 2))
        self.rand_neighbours     = None
        self.rand_D              = None
        self.anchors             = None  # points that are designated as anchor points for QA
        N_shepard_dists = min(5000, 3*N)
        self.shepard_distances1   = np.random.randint(size = N_shepard_dists, low = 0, high = N)
        self.shepard_distances2   = np.random.randint(size = N_shepard_dists, low = 0, high = N)
        self.initialised = False # set to True after the initialising threads are done

        initialiser_thread = threading.Thread(target=init_dataset, args=[self.X, self])
        initialiser_thread.start()

    def delete(self): # whoever points to Dataset should set the pointer to null and the memory is freed this way (can't delete self.perms or other since they can be used by other thread: they will be deleted once the other thread is done)
        self.deleted = True


def KNN_score(X, Y, is_classification):
    if not is_classification:
        from sklearn.neighbors import KNeighborsRegressor as KNN
    else:
        from sklearn.neighbors import KNeighborsClassifier as KNN

    N, M = X.shape
    N_train = int(0.65*N)
    perms = np.arange(N)
    score_acc = 0.
    n_trial   = 20
    for trial in range(n_trial):
        np.random.shuffle(perms)
        X = X[perms]
        Y = Y[perms]

        X_train, X_test = X[:N_train], X[N_train:]
        Y_train, Y_test = Y[:N_train], Y[N_train:]

        neigh = KNN(n_neighbors = 5)
        neigh.fit(X_train, Y_train)
        score_acc += neigh.score(X_test, Y_test)
    print("score : ", score_acc/n_trial)
