from DR_algorithms.DR_algorithm import DR_algorithm
import numpy as np
from sklearn.decomposition import PCA
from DR_algorithms.additional_DR_files.quartet_grads import compute_quartet_grads
import numba

def forward_pass(X_var, encoder_mds, encoder_offset, decoder):
    encoded_mds    = encoder_mds(X_var)
    encoded_offset = encoder_offset(X_var)
    embedding = encoded_mds + encoded_offset
    recon   = decoder(embedding)
    return encoded_mds, encoded_offset, embedding, recon

@numba.jit(nopython=True, fastmath = True)
def mds_iters_fast(Xhd, Xld_adjusted, perms2, batches_idxes, Dhd_quartet, grad_acc, n_iter, lr, momentums, momentum):
    beta = 1. - momentum
    for i in range(n_iter):
        for batch_idx in batches_idxes:
            quartet   = perms2[batch_idx]
            LD_points = Xld_adjusted[quartet]

            Dhd_quartet[0] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2))
            Dhd_quartet[1] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2))
            Dhd_quartet[2] = np.sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2))
            Dhd_quartet[3] = np.sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2))
            Dhd_quartet[4] = np.sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2))
            Dhd_quartet[5] = np.sqrt(np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2))

            Dhd_quartet  /= np.sum(Dhd_quartet)
            quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet)

            momentums[quartet[0], 0] = momentum*momentums[quartet[0], 0] + beta*quartet_grads[0]
            momentums[quartet[0], 1] = momentum*momentums[quartet[0], 1] + beta*quartet_grads[1]
            momentums[quartet[1], 0] = momentum*momentums[quartet[1], 0] + beta*quartet_grads[2]
            momentums[quartet[1], 1] = momentum*momentums[quartet[1], 1] + beta*quartet_grads[3]
            momentums[quartet[2], 0] = momentum*momentums[quartet[2], 0] + beta*quartet_grads[4]
            momentums[quartet[2], 1] = momentum*momentums[quartet[2], 1] + beta*quartet_grads[5]
            momentums[quartet[3], 0] = momentum*momentums[quartet[3], 0] + beta*quartet_grads[6]
            momentums[quartet[3], 1] = momentum*momentums[quartet[3], 1] + beta*quartet_grads[7]
        Xld_adjusted -= lr*momentums

def mds_iters(Xhd, Xld_adjusted, perms2, batches_idxes, Dhd_quartet, grad_acc, n_iter, lr, momentums, momentum):
    np.random.shuffle(perms2)
    mds_iters_fast(Xhd, Xld_adjusted, perms2, batches_idxes, Dhd_quartet, grad_acc, n_iter, lr, momentums, momentum)


class DL_method(DR_algorithm):
    def __init__(self, algo_name):
        super(DL_method, self).__init__(algo_name)
        self.add_int_hyperparameter('max epoch', 20, 10000, 10, 320)
        self.embedding = None

    def fit(self, progress_listener, X, Y):
        N, M = X.shape
        try:
            import torch
            from torch.autograd import Variable
        except:
            raise Exception("could not import pytorch.")

        hparams = self.get_hyperparameters()
        device = torch.device("cuda:0")

        encoder    = torch.nn.Sequential(torch.nn.Linear(M, int(4*M)), torch.nn.LeakyReLU(),\
                                         torch.nn.Linear(int(4*M), int(3*M)), torch.nn.LeakyReLU(),\
                                         torch.nn.Linear(int(3*M), int(2*M)), torch.nn.LeakyReLU(),\
                                         torch.nn.Linear(int(2*M), 8), torch.nn.Sigmoid(),\
                                         torch.nn.Linear(8, 2)\
                                        ).to(device)
        decoder = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.LeakyReLU(),\
                                         torch.nn.Linear(16, int(2*M)), torch.nn.LeakyReLU(),\
                                         torch.nn.Linear(int(2*M), int(3*M)), torch.nn.LeakyReLU(),\
                                         torch.nn.Linear(int(3*M), int(4*M)), torch.nn.LeakyReLU(),\
                                         torch.nn.Linear(int(4*M), M)\
                                        ).to(device)

        encoder.train(); decoder.train()

        optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()))

        max_epoch = hparams["max epoch"]
        X_train    = X.copy()

        kl_loss = torch.nn.KLDivLoss()
        N_train = X_train.shape[0]
        minibatch_size = 300
        batch_size = 2*minibatch_size
        perms = np.arange(N_train)
        for epoch in range(max_epoch):
            print('do on nearest neighbours ', epoch)
            np.random.shuffle(perms)
            for i, L in enumerate(range(0, N_train - batch_size + 1, batch_size)):
                M = L+minibatch_size
                R = M+minibatch_size
                B1_indices = perms[L:M]
                B2_indices = perms[M:R]
                B1 = Variable(torch.Tensor(X_train[B1_indices]), requires_grad=False).to(device)
                B2 = Variable(torch.Tensor(X_train[B2_indices]), requires_grad=False).to(device)

                total_loss = None
                L = 4
                for l in range(L):

                    lamda = np.random.uniform()
                    target = lamda*B1 + (1-lamda)*B2

                    B1_enc = encoder(B1)
                    B2_enc = encoder(B2)
                    mixed_enc = lamda*B1_enc + (1-lamda)*B2_enc
                    decoded = decoder(mixed_enc)

                    loss_recon = torch.mean((decoded-target)**2)
                    if total_loss is None:
                        total_loss = loss_recon
                    else:
                        total_loss += loss_recon

                        # d0 = torch.distributions.Normal(torch.zeros_like(mixed_enc[0]), torch.ones_like(mixed_enc[0]))
                        # p0 = d0.sample_n(minibatch_size)
                        #
                        # real_p0 = mixed_enc
                        # # print(kl_loss(p0, real_p0))
                        # total_loss += kl_loss(p0, real_p0)

                total_loss /= L
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                full  =  Variable(torch.Tensor(X), requires_grad=False).to(device)
                embedding = encoder(full)
                embedding = embedding.detach().to("cpu").numpy()
                progress_listener.notify((self.dataset_name, self.proj_name, embedding, self), [])

        encoder.eval(); decoder.eval()
        full  =  Variable(torch.Tensor(X), requires_grad=False).to(device)
        self.embedding = encoder(full).detach().to("cpu").numpy()

    # def fit(self, progress_listener, X, Y):
    #     N, M = X.shape
    #     try:
    #         import torch
    #         from torch.autograd import Variable
    #     except:
    #         raise Exception("could not import pytorch.")
    #
    #     hparams = self.get_hyperparameters()
    #     device = torch.device("cuda:0")
    #
    #     encoder_mds    = torch.nn.Sequential(torch.nn.Linear(M, int(3*M)), torch.nn.LeakyReLU(),\
    #                                     torch.nn.Linear(int(3*M), int(1.2*M)), torch.nn.LeakyReLU(),\
    #                                      # torch.nn.Linear(int(3*M), int(2*M)), torch.nn.LeakyReLU(),\
    #                                      # torch.nn.Linear(int(2*M), int(0.9*M)), torch.nn.LeakyReLU(),\
    #                                      torch.nn.Linear(int(1.2*M), 2)\
    #                                     ).to(device)
    #     encoder_offset = torch.nn.Sequential(torch.nn.Linear(M, int(3*M)), torch.nn.LeakyReLU(),\
    #                                      torch.nn.Linear(int(3*M), int(0.9*M)), torch.nn.LeakyReLU(),\
    #                                      torch.nn.Linear(int(0.9*M), 2)\
    #                                     ).to(device)
    #     decoder = torch.nn.Sequential(torch.nn.Linear(2, 6), torch.nn.LeakyReLU(),\
    #                                      torch.nn.Linear(6, 24), torch.nn.LeakyReLU(),\
    #                                      torch.nn.Linear(24, 42), torch.nn.LeakyReLU(),\
    #                                      torch.nn.Linear(42, int(2*M)), torch.nn.LeakyReLU(),\
    #                                      torch.nn.Linear(int(2*M), M)
    #                                     ).to(device)
    #
    #     encoder_mds.train(); encoder_offset.train(); decoder.train()
    #
    #     optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder_offset.parameters()) + list(encoder_mds.parameters()))
    #     optimizer_tmp = torch.optim.Adam(list(encoder_mds.parameters()))
    #
    #     max_epoch = hparams["max epoch"]
    #     X_train    = X.copy()
    #
    #     # MDS embedding
    #     Xld = PCA(n_components=2, whiten=True, copy=True).fit_transform(X).astype(np.float64)
    #     Xld *= 10/np.std(Xld)
    #     perms         = np.arange(N)
    #     batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
    #     mds_grad_acc  = np.ones((N, 2))
    #     Dhd_quartet   = np.zeros((6,))
    #     mds_momentums = np.zeros((N,2))
    #
    #
    #
    #     N_train    = X_train.shape[0]
    #     mds_lr = 200.
    #     mds_momentum_init = 0.999
    #     batch_size = 312
    #     perms = np.arange(N_train)
    #     perms2 = np.arange(N_train)
    #
    #     # mm = 2000
    #     # for i in range(mm):
    #     #     print(i)
    #     #     mds_momentum = min(mds_momentum_init, max(0.111, mds_momentum_init*(2/(1+np.exp(-(70*(i/mm)**3)))) -1))
    #     #     mds_iters(X, Xld, np.arange(N_train), batches_idxes, Dhd_quartet, mds_grad_acc, 2, mds_lr, mds_momentums, mds_momentum)
    #     # self.embedding = Xld
    #     # return
    #
    #     for epoch in range(120):
    #         mds_momentum = min(mds_momentum_init, max(0.111, mds_momentum_init*(2/(1+np.exp(-(70*(epoch/120)**3)))) -1))
    #         print('epoch', epoch)
    #         np.random.shuffle(perms)
    #         for i, L in enumerate(range(0, N_train - batch_size + 1, batch_size)):
    #             R = L+batch_size
    #             batch_indices = perms[L:R]
    #             batch = torch.Tensor(X_train[batch_indices]).to(device)
    #             X_var = Variable(batch, requires_grad=False).to(device)
    #             encoded_mds    = encoder_mds(X_var)
    #             with torch.no_grad():
    #                 Xld[batch_indices] = encoded_mds.detach().to("cpu").numpy()
    #                 mds_iters(X, Xld, perms2, batches_idxes, Dhd_quartet, mds_grad_acc, n_iter = 1, lr=mds_lr, momentums=mds_momentums, momentum=mds_momentum)
    #             mds_target = torch.Tensor(Xld[batch_indices]).to(device)
    #             loss_mds = torch.mean(torch.abs(encoded_mds-mds_target))
    #
    #             optimizer_tmp.zero_grad()
    #             loss_mds.backward()
    #             optimizer_tmp.step()
    #
    #     for epoch in range(max_epoch):
    #         mds_momentum = min(mds_momentum_init, max(0.111, mds_momentum_init*(2/(1+np.exp(-(70*(epoch/max_epoch)**3)))) -1))
    #         print('epoch', epoch)
    #         np.random.shuffle(perms)
    #         for i, L in enumerate(range(0, N_train - batch_size + 1, batch_size)):
    #             R = L+batch_size
    #             batch_indices = perms[L:R]
    #             batch = torch.Tensor(X_train[batch_indices]).to(device)
    #             X_var = Variable(batch, requires_grad=False).to(device)
    #
    #             encoded_mds, encoded_offset, embedding, recon = forward_pass(X_var, encoder_mds, encoder_offset, decoder)
    #
    #             with torch.no_grad():
    #                 # Xld[batch_indices] = encoded_mds.detach().to("cpu").numpy()
    #                 Xld[batch_indices] = (embedding-encoded_offset).detach().to("cpu").numpy()
    #                 mds_iters(X, Xld, perms2, batches_idxes, Dhd_quartet, mds_grad_acc, n_iter = 1, lr=mds_lr, momentums=mds_momentums, momentum=mds_momentum)
    #
    #             mds_target = torch.Tensor(Xld[batch_indices]).to(device)
    #             loss_mds = torch.mean(torch.abs(encoded_mds-mds_target))
    #             loss_recon = torch.mean(torch.abs(X_var-recon))
    #             loss_offset = torch.mean(torch.abs(torch.std(encoded_offset) - 2*1e-4))
    #
    #             # total_loss = 0.75*loss_mds + 0.25*loss_recon
    #             # total_loss = loss_mds + loss_recon + loss_offset
    #             # total_loss = loss_recon + 0.1*loss_offset + loss_mds
    #
    #             optimizer.zero_grad()
    #             total_loss.backward()
    #             optimizer.step()
    #
    #         if epoch % 10 == 0:
    #             full  =  torch.Tensor(X).to(device)
    #             X_var = Variable(full, requires_grad=False).to(device)
    #             encoded_mds, encoded_offset, embedding_final, recon = forward_pass(X_var, encoder_mds, encoder_offset, decoder)
    #             embedding_final = embedding_final.detach().to("cpu").numpy()
    #             # embedding_final = encoded_mds.detach().to("cpu").numpy()
    #             progress_listener.notify((self.dataset_name, self.proj_name, embedding_final, self), [])
    #
    #     encoder_mds.eval(); encoder_offset.eval(); decoder.eval()
    #     X_var = Variable(full, requires_grad=False).to(device)
    #     encoded_mds, encoded_offset, embedding_final, recon = forward_pass(X_var, encoder_mds, encoder_offset, decoder)
    #     self.embedding = embedding_final.detach().to("cpu").numpy()

    def transform(self, X, Y):
        return self.embedding


def build_encoder(M, n_hidden_before_latent, h1_sz, h2_sz, h3_sz, latent_activation_type):
    import torch
    if n_hidden_before_latent == 1:
        if latent_activation_type == "sigmoid":
            return torch.nn.Sequential(\
                                        torch.nn.Linear(M, h1_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h1_sz, 2), torch.nn.Sigmoid()\
                                    )
        else:
            return torch.nn.Sequential(\
                                        torch.nn.Linear(M, h1_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h1_sz, 2), torch.nn.LeakyReLU(0.05)\
                                    )
    elif n_hidden_before_latent == 2:
        if latent_activation_type == "sigmoid":
            return torch.nn.Sequential(\
                                        torch.nn.Linear(M, h1_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h1_sz, h2_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h2_sz, 2), torch.nn.Sigmoid()\
                                    )
        else:
            return torch.nn.Sequential(\
                                        torch.nn.Linear(M, h1_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h1_sz, h2_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h2_sz, 2), torch.nn.LeakyReLU(0.05)\
                                    )
    else:
        if latent_activation_type == "sigmoid":
            return torch.nn.Sequential(\
                                        torch.nn.Linear(M, h1_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h1_sz, h2_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h2_sz, h3_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h3_sz, 2), torch.nn.Sigmoid()\
                                    )
        else:
            return torch.nn.Sequential(\
                                        torch.nn.Linear(M, h1_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h1_sz, h2_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h2_sz, h3_sz), torch.nn.LeakyReLU(0.05),\
                                        torch.nn.Linear(h3_sz, 2), torch.nn.LeakyReLU(0.05)\
                                    )

def build_decoder(M, n_hidden_before_latent, h1_sz, h2_sz, h3_sz, latent_activation_type):
    import torch
    if n_hidden_before_latent == 1:
        return torch.nn.Sequential(\
                                    torch.nn.Linear(2, h1_sz), torch.nn.LeakyReLU(0.05),\
                                    torch.nn.Linear(h1_sz, M)\
                                )
    elif n_hidden_before_latent == 2:
        return torch.nn.Sequential(\
                                    torch.nn.Linear(2, h1_sz), torch.nn.LeakyReLU(0.05),\
                                    torch.nn.Linear(h1_sz, h2_sz), torch.nn.LeakyReLU(0.05),\
                                    torch.nn.Linear(h2_sz, M)\
                                )
    else:
        return torch.nn.Sequential(\
                                    torch.nn.Linear(2, h1_sz), torch.nn.LeakyReLU(0.05),\
                                    torch.nn.Linear(h1_sz, h2_sz), torch.nn.LeakyReLU(0.05),\
                                    torch.nn.Linear(h2_sz, h3_sz), torch.nn.LeakyReLU(0.05),\
                                    torch.nn.Linear(h3_sz, M)\
                                )
