
import numpy as np
import numba
from scipy.stats import zscore
from utils import honourable_death


class Dataset_loader():

    def __init__(self, config):
        self.default_N  = int(config["default_N"])

    def get(self, name):
        get_func = None
        N        = self.default_N
        if name == "winequality (red)":
            get_func = get_winequality
        elif name == "abalone":
            get_func = get_abalone
        elif name == "airfoil self-noise":
            get_func = get_airfoil
        elif name == "satellite":
            get_func = get_satellite
        elif name == "coil-20":
            get_func = get_coil20
        elif name == "coil-20 7% missing values":
            get_func = get_coil20_missing_values_7
        elif name == "blob 25":
            get_func = get_blob25
        elif name == "swissroll":
            get_func = get_swissroll
        elif name == "s-curve":
            get_func = get_Scurve
        elif name == "s-curve 5k":
            get_func = get_Scurve
            N = 5000
        elif name == "double-S":
            get_func = get_doubleS
            N = 3000
        elif name == "double-S 10k":
            get_func = get_doubleS
            N = 10000
        elif name == "bundled coils":
            get_func = get_bundled_coils
            N = 2500
        elif name == "spread coils":
            get_func = get_spread_coils
            N = 2500
        elif name == "Hastie 10":
            get_func = get_Hastie10
            N = 3000
        elif name == "D digits 8x8":
            get_func = D_get_digits8x8
        elif name == "digits 8x8":
            get_func = get_digits8x8
        elif name == "forest covtype":
            get_func = get_forest_covtype
            N = 3000
        elif name == "housing (download if not cached)":
            get_func = get_housing
            N = 2500
        elif name == "RNA-seq 3k":
            get_func = get_RNAseq
            N = 3000
        elif name == "RNA-seq 10k":
            get_func = get_RNAseq
            N = 10000
        elif name == "coil-20 37% missing values":
            get_func = get_coil20_missing_values_37
        elif name == "gestures":
            get_func = get_gestures
        elif name == "plant":
            get_func = get_plant
        elif name == "anuran":
            get_func = get_anuran
        else:
            honourable_death("unrecognised dataset : "+str(name))

        dataset_params = {
                        'is_dists' : False,
                        'X' : None,
                        'Y' : None,
                        'is_classification' : True,
                        'colors' : None,  # if kept at None: the colors are automatically generated and assigned to labels. regression datasets ignore this parameter
        }

        get_func(N, dataset_params) # fills the dataset_params dictionary

        X                    = dataset_params['X']
        Y                    = dataset_params['Y']
        is_labeled           = Y is not None
        is_classification    = dataset_params['is_classification']
        colors               = dataset_params['colors']

        # ~~~~~~~~~~ shuffle ~~~~~~~~~~
        if not dataset_params['is_dists']:
            permutation = np.arange(X.shape[0])
            np.random.shuffle(permutation)
            X = X[permutation]
            if is_labeled:
                Y = Y[permutation]

        # ~~~~~~~~~~ Y format ~~~~~~~~~~
        if is_labeled:
            if is_classification:
                Y = Y.astype(int)
                Y = Y.reshape((-1,))
                Y -= np.min(Y) # just in case the labels arent starting by 0
            else:
                Y = Y.astype(np.float64)

        # ~~~~~~~~~~ check if missing values ~~~~~~~~~~
        if ((not dataset_params['is_dists']) and np.isnan(np.min(X))):
            X = self.fill_missing_values(X)

        # X, Y = unsupervised_selection(X, Y, 4, is_classification)

        return X, Y, is_labeled, is_classification, colors, dataset_params['is_dists']


    # fills the locations in X where the value is np.nan
    # N_models: how many random feature sampling will be done, each will be used to guess missing values using KNN
    # how many features are used for each sampling. a lot of missing values require a smaller number here (we want to have a reasonable amount of points with a value present in all the sampled features)
    def fill_missing_values(self, X_with_missing, N_models=9000, prev_N_missing=None):
        from sklearn.neighbors import KDTree
        from scipy.spatial.distance import pdist
        # return fill_missing_values_simple(X_with_missing)

        N, M = X_with_missing.shape
        mask_x, mask_y = np.where(np.isnan(X_with_missing))
        mask = np.zeros_like(X_with_missing, dtype=bool)
        mask[mask_x, mask_y] = True
        if prev_N_missing is None:
            prev_N_missing = np.where(mask.ravel() == True)[0].shape[0]
        pct_missing = prev_N_missing/(N*M)


        N_features = find_target_N_features(X_with_missing, N, M) # aim for about 2% of the population for each model
        print("missing values: ", np.round(pct_missing, 3), " n features per sample: ", N_features, " n samples ", N_models)

        X = X_with_missing.copy().astype(np.float64)
        models = np.empty((N_models, N_features), dtype=int)
        for m in range(N_models):
            models[m] = np.random.choice(np.arange(M), size = N_features, replace = False)

        model_preds        = np.zeros_like(X, dtype=np.float64) # accumulates the predictions
        model_preds_counts = np.zeros_like(X, dtype= np.float64)       # counter used to compute the mean prediction
        knn_simil          = np.zeros((N, N))

        D_std = np.mean(np.nanstd(X, axis=0))

        import time
        tic = time.time()
        max_sec = 30
        for m in range(N_models):
            if m > 10 and int(time.time() - tic) >= max_sec:
                print("ending imputation: 30 seconds were elapsed (", m," iterations were done)")
                break

            if m % 500 == 0:
                print(np.round(100*m/N_models, 2), "%")
            model = models[m] # something like [3, 12, 0, 5] <-- indices of the features used by this model
            elligibles = np.where(~np.isnan(np.sum(X[:, model], axis=1)))[0] # observations where each of the observed features are present
            N_elligibles = elligibles.shape[0]
            if N_elligibles < 4:
                continue
            model_dataset = X[elligibles][:,model] # only look at the model's features for the elligible points
            # assert np.where(np.isnan(model_dataset))[0].shape[0] == 0 # check


            knn = KDTree(model_dataset).query(model_dataset, 2, return_distance=False)[:, 1:] # 1-NN in the elligible points
            compute_knn_simil(knn_simil, N_elligibles, elligibles, knn, X, model_dataset, D_std)
            knn_predictions(N_elligibles, elligibles, knn, X, model_preds, model_preds_counts, knn_simil) # for each mising value in the elligibles (all the features this time): the 1-NN gives its value if not a nan

        # model_preds_counts[np.where(model_preds_counts < 0.001)] = 0.
        all_missing_values_done = np.min(model_preds_counts[mask]) != 0.

        mask[np.where(model_preds_counts == 0.)] = False
        X[mask] = model_preds[mask] / model_preds_counts[mask]
        if not all_missing_values_done:
            print("not all values were found, the remaining are set with the mean feature value")
            mask_x, mask_y = np.where(np.isnan(X))
            mask = np.zeros_like(X, dtype=bool)
            mask[mask_x, mask_y] = True
            remaining = np.where(mask.ravel() == True)[0].shape[0]
            print("remaining : ", remaining/(N*M))
            return fill_missing_values_simple(X)
        return X

def find_target_N_features(X, N, M):
    n_elligibles = 0.
    target_pop_per_model = int(0.06 * N)
    pop_per_model = 0
    min_N, max_N = 2, max(2, M-4)
    L, R = min_N, max_N
    N_features   = int((L + R)/2)
    while pop_per_model < (target_pop_per_model - int(0.1*target_pop_per_model)) or pop_per_model > (target_pop_per_model + int(0.1*target_pop_per_model)):

        # print(N_features, "  ---", L, R)
        if N_features <= L:
            return L
        elif N_features >= R:
            return R

        pop_per_model = 0
        for i in range(20):
            model = np.random.choice(np.arange(M), size = N_features, replace = False)
            elligibles = np.where(~np.isnan(np.sum(X[:, model], axis=1)))[0]
            N_elligibles = elligibles.shape[0]
            pop_per_model += N_elligibles
        pop_per_model = int(pop_per_model/20)

        if pop_per_model < (target_pop_per_model - int(0.1*target_pop_per_model)):
            R = N_features
            N_features = int(0.5 * (N_features+L) - 1)
        elif pop_per_model > (target_pop_per_model + int(0.1*target_pop_per_model)):
            L = N_features
            N_features = int(1 + 0.5 * (N_features+R))
    return N_features




@numba.jit(nopython=True)
def get_means(X_with_missing):
    N, M = X_with_missing.shape
    means                = np.zeros((M,))
    contibution_counters = np.zeros((M,))
    for i in range(N):
        for j in range(M):
            if not np.isnan(X_with_missing[i, j]):
                means[j] += X_with_missing[i, j]
                contibution_counters[j] += 1
    means /= contibution_counters
    return means

@numba.jit(nopython=True)
def fill_missing_values_simple(X_with_missing):
    N, M = X_with_missing.shape
    means = get_means(X_with_missing)
    for i in range(N):
        for j in range(M):
            if np.isnan(X_with_missing[i, j]):
                X_with_missing[i, j] = means[j]
    return X_with_missing

@numba.jit(nopython=True)
def compute_knn_simil(knn_simil, N_elligibles, elligibles, knn, X, model_dataset, D_std):
    c = 0.31*((D_std)**2)
    for p in range(N_elligibles):
        my_idx    = elligibles[p]
        rel_neigh_idx = knn[p][0]
        neigh_idx     = elligibles[rel_neigh_idx]

        acc, cnt = 0., 1e-8
        for m in range(X.shape[1]):
            if not np.isnan(X[my_idx, m]) and not np.isnan(X[neigh_idx, m]):
                acc += (X[my_idx, m] - X[neigh_idx, m])**2
                cnt +=1
        d_sq = acc/cnt

        simil = max(1e-8, np.exp( -(d_sq) / c))
        knn_simil[my_idx, neigh_idx] = simil
        knn_simil[neigh_idx, my_idx] = simil



@numba.jit(nopython=True)
def knn_predictions(N_elligibles, elligibles, knn, X, model_preds, model_preds_counts, knn_simil):
    for p in range(N_elligibles):
        my_idx    = elligibles[p]
        rel_neigh_idx = knn[p][0]
        neigh_idx     = elligibles[rel_neigh_idx]

        simil = knn_simil[my_idx, neigh_idx]
        nan_locs  = np.where(np.isnan(X[my_idx]))[0]
        for loc in nan_locs:
            if not np.isnan(X[neigh_idx, loc]):
                model_preds[my_idx, loc]        += X[neigh_idx, loc]*simil
                model_preds_counts[my_idx, loc] += simil


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataset fetcher functions, as described at the beggining of this file.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

def get_RNAseq(N, dataset_params):
    if N == 3000:
        filename = 'datasets/RNAseq_N3k.npy'
    elif N == 10000:
        filename = 'datasets/RNAseq_N10k.npy'
    XY = np.load(filename)
    RNAcolors = np.load('datasets/RNAseq_colors.npy')

    rgb_colors = []
    for c in RNAcolors:
        rgb_colors.append(np.array(list(int(str(c).lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))) # taken from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb_colors = np.array(rgb_colors)

    X = XY[:, :-1]
    Y =  XY[:, -1]

    dataset_params['X'] = X
    dataset_params['Y'] =Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = rgb_colors


def get_doubleS(N, dataset_params):
    from sklearn import datasets
    (X, _) =  datasets.make_s_curve(n_samples=int(N/2), noise=0., random_state=None)
    Y = np.zeros((X.shape[0],))
    X = np.concatenate((X, X+np.array([0,0,3])))
    Y = np.concatenate((Y, np.ones((X.shape[0],))))
    dataset_params['X'] = zscore(X, axis=0)
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None

def get_housing(N, dataset_params):
    print("if this is the first time that the housing dataset is used, it needs to be downloaded. This can take a couple of minutes...")
    from sklearn import datasets
    (X, Y) =  datasets.fetch_california_housing(data_home=None, download_if_missing=True, return_X_y=True, as_frame=False)
    permutations = np.arange(X.shape[0])
    np.random.shuffle(permutations)
    X = X[permutations]
    Y = Y[permutations]
    X = X[:N]
    Y = Y[:N]
    dataset_params['X'] = zscore(X, axis=0)
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = False
    dataset_params['colors'] = None


def get_Hastie10(N, dataset_params):
    from sklearn import datasets
    (X, Y) =  datasets.make_hastie_10_2(n_samples=N)
    permutations = np.arange(X.shape[0])
    np.random.shuffle(permutations)
    Y[np.where(Y < 0)] = 0
    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None




def get_forest_covtype(N, dataset_params):
    from sklearn import datasets
    (X, Y) =  datasets.fetch_covtype(return_X_y=True, shuffle = True)
    X = X[:N]
    Y = Y[:N]
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None


def get_satellite(N, dataset_params):
    XY = np.genfromtxt('datasets/satellite.csv', delimiter=";", skip_header=1)
    Y = XY[:, -1] - 1
    X = XY[:, :-1]
    dataset_params['X'] = zscore(X, axis=0)
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None

def get_digits8x8(N, dataset_params):
    from sklearn import datasets
    (X, Y) =  datasets.load_digits(n_class=10, return_X_y=True, as_frame=False)
    X /= 16
    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None

def D_get_digits8x8(N, dataset_params):
    from sklearn import datasets
    (X, Y) =  datasets.load_digits(n_class=10, return_X_y=True, as_frame=False)
    X /= 16
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X)
    dataset_params['is_dists'] = True
    dataset_params['X'] = D
    dataset_params['D'] = D
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None


def get_spread_coils(N, dataset_params):
    nb_coils = 5
    from sklearn import datasets
    (X, _) =  datasets.make_swiss_roll(n_samples=int(N/nb_coils), noise=0.0, random_state=None)
    X_original = zscore(X, axis=0)
    X = X_original
    Y = np.zeros((X.shape[0],), dtype=int)
    for i in range(nb_coils-1):
        offset   = np.random.choice([-1, 1])*np.random.uniform(10, 30., size=(3,))
        rotation = np.random.random((X_original.shape[1], X.shape[1]))*3.14
        X = np.concatenate((X, np.dot(X_original, rotation)+offset))
        Y = np.concatenate((Y, (1+i)*np.ones((X_original.shape[0],))))
    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None


def get_bundled_coils(N, dataset_params):
    from sklearn import datasets
    nb_coils = 5
    (X, _) =  datasets.make_swiss_roll(n_samples=int(N/nb_coils), noise=0.0, random_state=None)
    X_original = zscore(X, axis=0)
    X = X_original
    Y = np.zeros((X.shape[0],), dtype=int)
    for i in range(nb_coils-1):
        offset   = np.random.choice([-0.7, 0.7])*np.random.uniform(0.3, 8., size=(3,))
        rotation = np.random.random((X_original.shape[1], X.shape[1]))*3.14
        X = np.concatenate((X, np.dot(X_original, rotation)+offset))
        Y = np.concatenate((Y, (1+i)*np.ones((X_original.shape[0],))))
    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None


def get_swissroll(N, dataset_params):
    from sklearn import datasets
    (X, _) =  datasets.make_swiss_roll(n_samples=N, noise=0.0, random_state=None)
    dataset_params['X'] = zscore(X, axis=0)
    dataset_params['Y'] = None
    dataset_params['is_classification'] = False
    dataset_params['colors'] = None

def get_Scurve(N, dataset_params):
    from sklearn import datasets
    (X, _) =  datasets.make_s_curve(n_samples=N, noise=0., random_state=None)
    dataset_params['X'] = zscore(X, axis=0)
    dataset_params['Y'] = None
    dataset_params['is_classification'] = False
    dataset_params['colors'] = None


def get_blob25(N, dataset_params):
    from sklearn import datasets
    (X, Y) =  datasets.make_blobs(n_samples=N, n_features=25, centers=5, \
            cluster_std=1.0, center_box=(-1.0, 1.0), shuffle=True, random_state=7, return_centers=False)
    X = zscore(X, axis=0)

    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None


def get_coil20(N, dataset_params):
    from scipy.io import loadmat
    mat = loadmat("datasets/COIL20.mat")
    X, Y = mat['X'], mat['Y']
    Y = (Y.astype(int) - 1).reshape((-1,))
    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None
    # dataset_params['supervised feature selection'] = True



def get_coil20_missing_values_7(N, dataset_params):
    get_coil20(N, dataset_params)
    X, Y = dataset_params['X'], dataset_params['Y']
    p = 0.07
    mask = np.random.choice([True, False], size=X.shape[0]*X.shape[1], p=[p, 1-p]).reshape(X.shape)
    X[mask] = np.nan
    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None

def get_coil20_missing_values_37(N, dataset_params):
    get_coil20(N, dataset_params)
    X, Y = dataset_params['X'], dataset_params['Y']
    p = 0.37
    mask = np.random.choice([True, False], size=X.shape[0]*X.shape[1], p=[p, 1-p]).reshape(X.shape)
    X[mask] = np.nan
    dataset_params['X'] = X
    dataset_params['Y'] = Y
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None


def get_abalone(N, dataset_params):
    XY = np.genfromtxt('datasets/abalone.csv', delimiter=",", skip_header=1)
    dataset_params['X'] = zscore(XY[:, :-1], axis=0)
    dataset_params['Y'] = XY[:, -1]
    dataset_params['is_classification'] = False
    dataset_params['colors'] = None



def get_airfoil(N, dataset_params):
    XY = np.genfromtxt('datasets/airfoil_noise.csv', delimiter=";", skip_header=1)
    dataset_params['X'] = zscore(XY[:, :-1], axis=0)
    dataset_params['Y'] = XY[:, -1]
    dataset_params['is_classification'] = False
    dataset_params['colors'] = None



def get_winequality(N, dataset_params):
    XY = np.genfromtxt('datasets/winequality-red.csv', delimiter=";", skip_header=1)
    X = zscore(XY[:, :-1], axis=0)
    dataset_params['X'] = X
    dataset_params['Y'] = XY[:, -1]
    dataset_params['is_classification'] = False # considered as a regression
    dataset_params['colors'] = None



def get_plant(N, dataset_params):
    import pandas as pd
    dataframe = pd.read_csv("./datasets/plant.csv")
    X = dataframe[['AT','V','AP','RH']].to_numpy().astype(np.float64)
    Y = dataframe[['PE']].to_numpy().astype(np.int)
    permutations = np.arange(X.shape[0])
    np.random.shuffle(permutations)
    X = X[permutations]
    Y = Y[permutations]
    X = X[:N]
    Y = Y[:N]
    dataset_params['X'] = zscore(X, axis=0)
    dataset_params['Y'] = Y.reshape((-1,))
    dataset_params['is_classification'] = False
    dataset_params['colors'] = None



def get_gestures(N, dataset_params):
    import pandas as pd
    str_to_class_nb = {'Hold':0, 'Preparation':1, 'Preparação':1, 'Rest':2, 'Retraction':3, 'Stroke':4}
    file = './datasets/gesture_full_raw.csv'
    df = pd.read_csv(file)
    vars = np.array(['lhx','lhy','lhz','rhx','rhy','rhz','hx','hy','hz','sx','sy','sz','lwx','lwy','lwz','rwx','rwy','rwz']).astype(str)
    label = ['Phase']
    X = df[vars].to_numpy().astype(np.float64)
    Y = df[label].to_numpy()
    for key in str_to_class_nb:
        Y[np.where(Y == key)] = str_to_class_nb[key]

    perms = np.arange(X.shape[0])
    np.random.shuffle(perms)
    X = X[perms]
    Y = Y[perms]
    dataset_params['X'] = zscore(X, axis=0)[:4000]
    dataset_params['Y'] = Y[:4000].reshape((-1,))
    dataset_params['is_classification'] = False
    dataset_params['colors'] = None


def get_anuran(N, dataset_params):
    import pandas as pd
    dataframe = pd.read_csv("./datasets/anuran.csv")

    dataframe.drop('RecordID',axis=1,inplace=True)
    dataframe.replace('Bufonidae', 0 ,inplace=True)
    dataframe.replace('Dendrobatidae',1,inplace=True)
    dataframe.replace('Hylidae',2,inplace=True)
    dataframe.replace('Leptodactylidae',3,inplace=True)

    dataframe.replace('Adenomera',0,inplace=True)
    dataframe.replace('Ameerega',1,inplace=True)
    dataframe.replace('Dendropsophus',2,inplace=True)
    dataframe.replace('Hypsiboas',3,inplace=True)
    dataframe.replace('Leptodactylus',4,inplace=True)
    dataframe.replace('Osteocephalus',5,inplace=True)
    dataframe.replace('Rhinella',6,inplace=True)
    dataframe.replace('Scinax',7,inplace=True)

    dataframe.replace('AdenomeraAndre',0,inplace=True)
    dataframe.replace('AdenomeraHylaedactylus',1,inplace=True)
    dataframe.replace('Ameeregatrivittata',2,inplace=True)
    dataframe.replace('HylaMinuta',3,inplace=True)
    dataframe.replace('HypsiboasCinerascens',4,inplace=True)
    dataframe.replace('HypsiboasCordobae',5,inplace=True)
    dataframe.replace('LeptodactylusFuscus',6,inplace=True)
    dataframe.replace('OsteocephalusOophagus',7,inplace=True)
    dataframe.replace('Rhinellagranulosa',8,inplace=True)
    dataframe.replace('ScinaxRuber',9,inplace=True)
    X, Y = dataframe[['MFCCs_ 1','MFCCs_ 2','MFCCs_ 3','MFCCs_ 4','MFCCs_ 5','MFCCs_ 6','MFCCs_ 7','MFCCs_ 8','MFCCs_ 9','MFCCs_10','MFCCs_11','MFCCs_12','MFCCs_13','MFCCs_14','MFCCs_15','MFCCs_16','MFCCs_17','MFCCs_18','MFCCs_19','MFCCs_20','MFCCs_21','MFCCs_22']].to_numpy().astype(np.float32), dataframe[['Genus']].to_numpy().astype(np.int)
    X = zscore(X, axis=0)
    perms = np.arange(X.shape[0])
    np.random.shuffle(perms)
    X = X[perms]
    Y = Y[perms]

    dataset_params['X'] = X[:4000]
    dataset_params['Y'] = Y[:4000].reshape((-1,))
    dataset_params['is_classification'] = True
    dataset_params['colors'] = None


    return X, Y.reshape((-1,)), True, None
