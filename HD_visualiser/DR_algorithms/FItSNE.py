from DR_algorithms.DR_algorithm import DR_algorithm

'''
as implemented in https://github.com/KlugerLab/FIt-SNE
by the authors:
George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, Yuval Kluger. (2019). Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. Nature Methods.
'''
class FItSNE(DR_algorithm):
    def __init__(self, algo_name):
        super(FItSNE, self).__init__(algo_name)
        self.add_string_hyperparameter('method', ["FItSNE","hybrid","SQuaD_MDS"], "hybrid")
        self.add_int_hyperparameter('n iter', 100, 50000, 100, 700)
        self.embedding = None

    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")
        try:
            from DR_algorithms.FItSNE_files.fast_tsne import fast_tsne
            # from .FLT.fast_tsne import fast_tsne # import from the file directly. place the contents of the github directly in a file called 'FLT' to make this import work
        except:
            raise Exception("could not import FItSNE. Modify the import in fit() in FItSNE.py to solve the issue. If you want to use the files from the author's github (https://github.com/KlugerLab/FIt-SNE) directly, then place the github files in a folder called 'FLT' inside the folder 'DR_algorithms' and change the import in the try statement above.")
        hparams = self.get_hyperparameters()
        N, n_PC = X.shape
        if hparams['method'] == "hybrid":
            X_LD = fast_tsne(X, method_type=hparams['method'], perplexity_list = [4, 50], max_iter = hparams['n iter'], early_exag_coeff=2.)
        else:
            X_LD = fast_tsne(X, method_type=hparams['method'], perplexity_list = [4, 50], max_iter = hparams['n iter'])

        self.embedding = X_LD
        super(FItSNE, self).compute_stress(X, " FItSNE ")

    def transform(self, X, Y):
        return self.embedding
