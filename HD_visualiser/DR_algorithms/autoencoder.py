from DR_algorithms.DR_algorithm import DR_algorithm
import numpy as np

class autoencoder(DR_algorithm):
    def __init__(self, algo_name):
        super(autoencoder, self).__init__(algo_name)
        self.add_string_hyperparameter('latent activation', ["leaky_relu", "sigmoid"], "sigmoid")
        self.add_int_hyperparameter('encoder hidden1 size', 5, 800, 1, 64)
        self.add_int_hyperparameter('encoder hidden2 size', 5, 400, 1, None)
        self.add_int_hyperparameter('encoder hidden3 size', 5, 100, 1, None)
        self.add_int_hyperparameter('max epoch', 20, 10000, 10, 500)
        self.add_float_hyperparameter('adam init lr', 1e-4, 1e-1, 1e-4, 1e-3)
        self.add_bool_hyperparameter('on gpu', True)
        self.add_float_hyperparameter('train size', 0.1, 1., 0.05, 1.)
        self.add_float_hyperparameter('batch size', 0.01, 0.5, 0.01, 0.1)
        self.add_bool_hyperparameter('show progress', True)
        self.embedding = None

    def fit(self, progress_listener, X, Y, is_dists):
        if is_dists:
            print("model need to be adapted for using distances as input matrix")
            raise Exception("dist matrix ")
        try:
            import torch
            from torch.autograd import Variable
        except:
            raise Exception("could not import pytorch. aborting autoencoder")
        hparams = self.get_hyperparameters()
        show_progress = (None not in [progress_listener, self.dataset_name, self.proj_name]) and hparams["show progress"]
        device = torch.device("cuda:0" if (hparams["on gpu"] and torch.cuda.is_available()) else "cpu")

        n_hidden_before_latent = 1
        if hparams['encoder hidden2 size'] is not None:
            n_hidden_before_latent += 1
        if n_hidden_before_latent==2 and hparams['encoder hidden3 size'] is not None:
            n_hidden_before_latent += 1

        encoder = build_encoder(X.shape[1], n_hidden_before_latent, hparams['encoder hidden1 size'],hparams['encoder hidden2 size'],hparams['encoder hidden3 size'], hparams["latent activation"]).to(device)
        decoder = build_decoder(X.shape[1], n_hidden_before_latent, hparams['encoder hidden1 size'],hparams['encoder hidden2 size'],hparams['encoder hidden3 size'], hparams["latent activation"]).to(device)
        encoder.train(); decoder.train()

        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=hparams['adam init lr'])

        max_epoch = hparams["max epoch"]
        X_train    = X[:int(X.shape[0] * hparams["train size"])].copy()
        N_train    = X_train.shape[0]
        batch_size = int(N_train * hparams["batch size"])
        perms = np.arange(N_train)
        for epoch in range(max_epoch):
            np.random.shuffle(perms)
            X_train = X_train[perms]
            for i, start in enumerate(range(0, N_train - batch_size + 1, batch_size)):
                batch = torch.Tensor(X_train[start:start+batch_size]).to(device)
                X_var = Variable(batch, requires_grad=False).to(device)

                encoded = encoder(X_var)
                recon   = decoder(encoded)

                loss_recon    = torch.mean((batch-recon)**2)

                optimizer.zero_grad()
                loss_recon.backward()
                optimizer.step()

            if show_progress and epoch % 10 == 0:
                full  =  torch.Tensor(X).to(device)
                X_var = Variable(full, requires_grad=False).to(device)
                encoded = encoder(X_var).detach().to("cpu").numpy()
                progress_listener.notify((self.dataset_name, self.proj_name, encoded, self), [])

        encoder.eval(); decoder.eval()
        full  =  torch.Tensor(X).to(device)
        X_var = Variable(full, requires_grad=False).to(device)
        encoded = encoder(X_var).detach().to("cpu").numpy()
        self.embedding = encoded

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
