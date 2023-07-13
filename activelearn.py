import os, shutil, pdb, sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import torch
from torch.utils.data import DataLoader, Dataset
import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal as mvn
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append('/mmfs1/home/kiranvad/kiranvad/neural-processes')
from neural_process import NeuralProcess
from training import NeuralProcessTrainer
from utils import context_target_split

# Define a GP (batch independent)
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([dim]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([dim])),
            batch_shape=torch.Size([dim])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

    def get_covaraince(self, x, xp):          
        cov = self.covar_module(x, xp).to_dense()
        K = cov.mean(axis=0).cpu().numpy().squeeze()

        return K


def train(i_query, n_tasks, data, time, np_model, n_iterations):
    # collect current z predictions
    # with torch.no_grad():
    t = torch.from_numpy(time.astype(np.float32))
    t = t.repeat(data.x.shape[0], 1).to(device)
    z, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2),
    data.y.unsqueeze(2))

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks, rank=0).to(device)
    model = GPModel(data.x, z, likelihood, n_tasks).to(device)

    # Setup model training
    model.train()
    likelihood.train()

    # Use the adam optimizer
    # Includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)  

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(data.x)
        # Calc loss and backprop gradients
        loss = -mll(output, z)
        loss.backward(retain_graph=True)

        optimizer.step()
        print('%d - loss : %.3f'%(i, loss.item()))

    return model, loss.item()

# Custom utility functions for active learning 

def predict(model, x):
    model.eval()
    with torch.no_grad():
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32).to(device)
        else:
            x = x.clone().detach()
        dist = model(x)

        return dist.mean.cpu().numpy(), dist.stddev.cpu().numpy()

def from_comp_to_spectrum(time, gp_model, np_model, c):
    with torch.no_grad():
        c = torch.tensor(c, dtype=torch.float32).to(device)
        z_sample,_ = predict(gp_model, c)
        z_sample = torch.tensor(z_sample, dtype=torch.float32).to(device)
        t = torch.from_numpy(time.astype(np.float32)).to(device)
        t = t.repeat(c.shape[0]).view(c.shape[0], len(time), 1)
        mu, std = np_model.xz_to_y(t, z_sample)

        return mu, std    

# Active learning functions
def generate_pool(sim, n_samples, RNG):
    # assemble initial data
    C_train = sim.points.astype(np.float32)
    y_train = np.asarray(sim.F, dtype=np.float32)
    initial_idx = RNG.choice(range(len(C_train)),
                            size=n_samples, 
                            replace=False
                            )
    C_initial = C_train[initial_idx,:]
    y_initial = y_train[initial_idx,:]
    colomap_indx = [0]*n_samples
    # generate the pool
    # remove the initial data from the training dataset
    C_pool = np.delete(C_train, initial_idx, axis=0)
    y_pool = np.delete(y_train, initial_idx, axis=0)

    return C_train, y_train, C_initial, y_initial, C_pool, y_pool, colomap_indx

def utility(model, C_query):
    _, sigma = predict(model, C_query)

    return sigma.mean(axis=1)

def selector(values, n_instances):
    max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
    return max_idx

def query_strategy(model, C_query, n_instances=1):
    u = utility(model, C_query)
    query_idx = selector(u, n_instances)

    return query_idx

class ActiveLearningDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = self.to_tensor(x,y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xs = self.x[idx]
        ys = self.y[idx]

        return xs, ys 

    def to_tensor(self, x, y):
        x_ = torch.tensor(x, dtype=torch.float32).to(device)
        y_ = torch.tensor(y, dtype=torch.float32).to(device)

        return x_, y_
    
    def update(self, x, y):
        x, y = self.to_tensor(x, y)
        self.x = torch.vstack((self.x, x))
        self.y = torch.vstack((self.y, y))

        return


# Neural Process model functions
class NPModelDataset(Dataset):
    def __init__(self, time, y):
        self.data = []
        for yi in y:
            xi = torch.from_numpy(time.astype(np.float32))
            xi = xi.view(xi.shape[0],1).to(device)
            yi = yi.view(yi.shape[0],1).to(device)
            self.data.append((xi,yi))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def update_npmodel(time, np_model, data):
    batch_size = 2
    num_context = 75
    num_target = 25
    dataset = NPModelDataset(time, data.y)
    data_loader = DataLoader(dataset, 
    batch_size=batch_size, shuffle=True)
    np_optimizer = torch.optim.Adam(np_model.parameters(), lr=3e-4)
    np_trainer = NeuralProcessTrainer(device, 
    np_model, np_optimizer,
    num_context_range=(num_context, num_context),
    num_extra_target_range=(num_target, num_target),
    verbose = False
    )

    np_model.training = True
    np_trainer.train(data_loader, 30)
    loss = np_trainer.epoch_loss_history[-1]
    print('NP model loss : %.2f'%loss)

    # freeze model training
    np_model.training = False

    return np_model, loss