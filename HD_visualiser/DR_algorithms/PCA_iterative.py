from DR_algorithms.DR_algorithm import DR_algorithm
from numba import njit
import numpy as np
import time

# @njit
def orthogonality_updates(components):
    Ncomp, M = components.shape
    updates = np.zeros_like(components)
    for i in range(Ncomp):
        for j in range(Ncomp):
            if i != j:
                towards = components[j] - components[i]
                towards_norm = np.sqrt(np.sum(towards*towards))
                towards_normalised = towards / towards_norm

                simi = np.sum(components[i] * components[j])
                updates[i] += simi * towards
    updates /= max(1, Ncomp-1)
    print(updates)
    print(updates.shape)
    print(components.shape)

    return updates


@njit
def nearby_components_updates(components, neighbour_components, comp_i):
    N_neighbours = neighbour_components.shape[0]

    component = components[comp_i]
    component_norm = np.sqrt(np.sum(component*component)) + 1e-12
    component_normalised = component / component_norm

    update = np.zeros(component.shape)
    for neighbour in range(N_neighbours):
        other_comp = neighbour_components[neighbour, comp_i]
        other_comp_normalised = other_comp/(np.sqrt(np.sum(other_comp*other_comp)) + 1e-12)
        simi = np.dot(component_normalised, other_comp_normalised)

        # if simi < 0.:
        #     simi     = -simi
        #     gradient = (1-simi) * -other_comp_normalised
        #     # gradient = (1-simi) * other_comp_normalised
        # else:
        #     gradient = (1-simi) * other_comp_normalised
        gradient = other_comp_normalised

        update += gradient
    return update / N_neighbours



@njit
def pca_iter(grad_norms, Xi_uncentered, components, ghosts, center, momentums, lr, neighbour_components, gradient_0 = True, residuals_long=False, L1_strength=0., alpha_neighbours=0., true_pca=False):
    Xi = Xi_uncentered - center
    Ncomp, M = components.shape
    sqrtM = np.sqrt(M)
    do_neighbour_grad = alpha_neighbours > 1e-6
    momentum_decay = 0.99
    momentums *= momentum_decay

    lr = 6*1e-3

    residual = Xi
    for comp in range(Ncomp):
        this_component = components[comp]
        residual_norm = np.sqrt(np.sum(residual*residual))
        if residual_norm < 1e-12:
            break

        simi = np.dot(residual / residual_norm, this_component)
        sign = 1.
        if simi < 0.:
            sign = -1.
            simi     = -simi
        gradient = sign * (1-simi) * residual_norm * residual
        # gradient = (1-simi) * residual

        grad_norm = np.sqrt(np.sum(gradient*gradient))
        grad_norms[comp] *= 0.99
        grad_norms[comp] += 0.01*grad_norm

        gradient /= grad_norm
        if not true_pca:
            gradient *= min(1., (grad_norm / grad_norms[comp])**2)
        else:
            gradient *= (grad_norm / grad_norms[comp])

        tmp_params = (this_component + momentums[comp]).reshape((M, 1)) # Nesterov momentums

        # tmp_params /= np.sqrt(np.sum(tmp_params*tmp_params))

        projected = np.dot(residual, tmp_params)
        reconstructed = np.dot(tmp_params, projected)
        residual -= reconstructed


        if do_neighbour_grad:
            grads_neighbours = nearby_components_updates(components, neighbour_components, comp)
            gradient *= (1 - alpha_neighbours)
            gradient += grads_neighbours * (alpha_neighbours)

        update_momentum = lr * gradient * (1-momentum_decay)
        momentums[comp] += update_momentum
        components[comp] += momentums[comp]


    if L1_strength > 1e-6:
        components /= np.sqrt(np.sum(components*components, axis=1)).reshape((-1, 1))
        added = components * L1_strength
        components += added
    components /= np.sqrt(np.sum(components*components, axis=1)).reshape((-1, 1))
    return components


@njit
def pca_iter_ND_batch(grad_norms, sample_uncentered, components, ghosts, center, momentums, lr, neighbour_components, gradient_0 = True, residuals_long=False, L1_strength=0., alpha_neighbours=0., true_pca=False):
    sample = sample_uncentered - center
    Ncomp, M = components.shape
    sqrtM = np.sqrt(M)
    sample_size = sample.shape[0]

    do_neighbour_grad = alpha_neighbours > 1e-6
    momentum_decay = 0.9
    momentums *= momentum_decay
    # momentums += orthogonality_updates(components) * lr * 0.0001

    grads_acc = np.zeros((Ncomp, M))
    residuals = sample
    for comp in range(Ncomp):
        this_component = components[comp]
        # residuals -= np.mean(residuals, axis=0)
        for pt in range(sample_size):

            residual_norm = np.sqrt(np.sum(residuals[pt]*residuals[pt]))
            if residual_norm < 1e-12:
                continue
            residual_normalised = residuals[pt] / residual_norm

            simi = np.dot(residual_normalised, this_component)
            if simi < 0.:
                residual_normalised = -residual_normalised
                simi     = -simi

            gradient = (1-simi) * residual_norm * (residual_norm * residual_normalised)
            # gradient = residual_norm * residual_norm * residuals[pt]

            grad_norm = np.sqrt(np.sum(gradient*gradient))
            grad_norms[comp] *= 0.99
            grad_norms[comp] += 0.01*grad_norm

            gradient /= grad_norm
            if not true_pca:
                gradient *= min(1., (grad_norm / grad_norms[comp])**2)
            else:
                gradient *= (grad_norm / grad_norms[comp])
            grads_acc[comp] += gradient

            tmp_params = (this_component + momentums[comp]).reshape((M, 1))
            # tmp_params /= np.sqrt(np.sum(tmp_params*tmp_params, axis=1)).reshape((-1, 1))
            if L1_strength > 1e-6:
                tmp_params += np.sqrt(np.abs(tmp_params))*tmp_params * L1_strength

            projected = np.dot(residuals[pt], tmp_params)
            reconstructed = np.dot(tmp_params, projected)
            residuals[pt] -= reconstructed

        grads_acc[comp] /= sample_size

        if do_neighbour_grad:
            grads_neighbours = nearby_components_updates(components, neighbour_components, comp)
            grads_acc[comp] *= (1 - alpha_neighbours)
            grads_acc[comp] += grads_neighbours * (alpha_neighbours)

        update_momentum = lr * grads_acc[comp] * (1-momentum_decay)

        momentums[comp] += update_momentum
        # norm_update_momentum = 1e-8 + update_momentum / np.sqrt(np.sum(update_momentum*update_momentum))
        # norm_momentum = 1e-8 + momentums[comp] / np.sqrt(np.sum(momentums[comp]*momentums[comp]))
        # momentums[comp] += update_momentum * (0.5 + np.abs(np.sum( (update_momentum/norm_update_momentum) * (momentums[comp]/norm_momentum)) ))
        # # print(np.abs(np.sum( (update_momentum/norm_update_momentum) * (momentums[comp]/norm_momentum)) ))
        # # momentums[comp] += update_momentum * 20 * (0.05 + np.abs(np.dot(update_momentum, momentums[comp])))

        components[comp] += momentums[comp]


    if L1_strength > 1e-6:
        components /= np.sqrt(np.sum(components*components, axis=1)).reshape((-1, 1))
        added = 1e-6+ np.sqrt(np.sqrt(np.abs(components)))*np.sqrt(np.abs(components))*components * L1_strength
        components += added
    components /= np.sqrt(np.sum(components*components, axis=1)).reshape((-1, 1))
    return components

@njit
def transform(X_uncentered, center, components, N_components=-1, compute_var=True, centered=False):
    if N_components == -1:
        N_components = components.shape[0]

    N, M = X_uncentered.shape
    Xld = np.zeros((N, N_components))
    if not centered:
        X = X_uncentered - center
    residuals = X
    for comp in range(N_components):
        # residuals -= np.mean(residuals, axis=0)
        for pt in range(N):
            projected = np.dot(residuals[pt], (components[comp]).reshape((1, M)).T)
            reconstructed = np.dot((components[comp]).reshape((1, M)).T, projected)
            residuals[pt] -= reconstructed
            Xld[pt, comp] = projected[0]
    return Xld


class PCA_iterative(DR_algorithm):
    def __init__(self, algo_name):
        super(PCA_iterative, self).__init__(algo_name)
        self.add_int_hyperparameter('niter', 10, 1000, 10, 200)
        self.add_int_hyperparameter('Mpca', 2, 50, 1, 2)
        self.add_float_hyperparameter('alpha neighbours', 0.00, 1., 0.05, 0.)
        self.add_bool_hyperparameter('hierarchical', True)
        self.add_bool_hyperparameter('residuals long', False)
        self.add_bool_hyperparameter('gradient with simi', False)
        self.add_float_hyperparameter('lr', 0.001, 1., 0.005, 0.25)
        self.add_float_hyperparameter('L1 strength', 0., 0.2, 0.025, 0.)
        self.add_float_hyperparameter('decay', 0.99, 1., 0.0001, 1.)
        self.model = None

    def update_the_screen(self, progress_listener, X, components):
        tic = time.time()
        # progress_listener.notify((self.dataset_name, self.proj_name, np.dot(X, components[:2].T), self), [])
        progress_listener.notify((self.dataset_name, self.proj_name, transform(X, np.mean(X,axis=0), components)[:,:2], self), [])
        toc = time.time()
        time.sleep(toc-tic+0.02)

    def plot_variance(self, X, center, components):
        import matplotlib.pyplot as plt
        Mpca, M = components.shape
        vars, vars2 = [], []
        import sklearn.decomposition
        pca_full = sklearn.decomposition.PCA(n_components=M,copy=True)
        pca_full.fit(X)
        Xmyalgo   = transform(X, center, components)
        Xpca_full = np.dot(X, pca_full.components_.T)
        vars_pca     = np.var(Xpca_full, axis=0)
        vars_my_algo = np.var(Xmyalgo, axis=0)
        var0 = np.sum(vars_pca)
        # plt.plot(np.cumsum(vars_pca) / var0)
        # plt.plot(np.cumsum(vars_my_algo) / var0)
        # plt.show()
        print(np.cumsum(vars_pca) / var0 , " PCA ")
        print(np.cumsum(vars_my_algo) / var0, " MINE")

    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")
        hparams = self.get_hyperparameters()
        N, M = X.shape
        perms = np.arange(N)
        Mpca = hparams['Mpca']
        hierarchical = hparams['hierarchical']
        gradient_0 = hparams['gradient with simi']
        lr = hparams['lr']
        niter = hparams['niter']
        decay = hparams['decay']
        residuals_long = hparams['residuals long']
        L1_strength = hparams['L1 strength']
        alpha_neighbours = hparams['alpha neighbours']

        components = np.random.uniform(size=(Mpca, M))
        components /= np.sqrt(np.sum(components**2, axis=1))[:, None]
        ghosts = components.copy()
        momentums = np.zeros_like(components)
        grad_norms = np.zeros((Mpca))

        neighbour_components = np.zeros((3, components.shape[0], components.shape[1]))
        neighbour_components[0] = -components.copy()
        neighbour_components[1] = -components.copy()
        neighbour_components[2] = -components.copy()
        neighbour_components *= -1

        center = np.mean(X, axis=0)

        for epoch in range(niter):
            if decay < 1.:
                lr *= decay
            sample = np.random.choice(np.arange(N), size=50, replace=False)
            for Xi in X[sample]:
                pca_iter(grad_norms, Xi, components, ghosts, center = center, momentums=momentums, neighbour_components=neighbour_components, lr=lr, gradient_0=gradient_0, residuals_long=residuals_long, L1_strength=L1_strength, alpha_neighbours=alpha_neighbours)
            # pca_iter_ND_batch(grad_norms, X[sample], components, ghosts, center = center, momentums=momentums, neighbour_components=neighbour_components, lr=lr, gradient_0=gradient_0, residuals_long=residuals_long, L1_strength=L1_strength, alpha_neighbours=alpha_neighbours)
            if N < 5000:
                self.update_the_screen(progress_listener, X, components)
            else:
                if epoch % 10 == 0:
                    self.update_the_screen(progress_listener, X, components)



        self.plot_variance(X, center, components)

        # print(np.sqrt(np.sum(components*components)), np.sqrt(np.sum(comp2*comp2)))
        # print(np.sum(components*components), np.sum(comp2*comp2))

        self.embedding = transform(X, np.mean(X,axis=0), components)[:,:2]

    '''
     missing value imputation en parcourant l'arbre de PCA
    '''


    def transform(self, X, Y):
        return self.embedding
