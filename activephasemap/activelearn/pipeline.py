import os, shutil, pdb, sys

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal as mvn
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


