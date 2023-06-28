import os, shutil, pdb, sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import torch
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gpytorch

sys.path.append(['./simulators', './plot', './activelearn'])
from simulators import *
from plot import *
from activelearn import *

SAVE_DIR = './dklgp/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

BATCH_SIZE = 2
RNG = np.random.default_rng()
N_INITIAL = 5 # number of initial points for AL
N_QUERIES = 100 # Total number of queries to simulator/oracle
N_GP_ITERATIONS = 200 # number of GP fitting iterations
N_LATENT = 3 
N_COMPOSITION = 2

# Define the simulator 
sim = GaussianPhases(n_grid=100, use_random_warping=False, noise=True)
sim.generate()
time = sim.t
sim.plot(SAVE_DIR+'phasemap.png')

# Specify the Neural Process model
np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load('./pretrain/trained_model.pt', map_location=device))

C_train, y_train, C_initial, y_initial, C_pool, y_pool, colomap_indx = generate_pool(sim, N_INITIAL,RNG)

# Define DKL-GP
class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, dim):
        super().__init__()
        self.add_module('linear1', torch.nn.Linear(dim, 32))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(32, 16))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(16, 8))
        self.add_module('relu3', torch.nn.ReLU())                
        self.add_module('linear4', torch.nn.Linear(8, 2))

class GPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, z_dim):
            super().__init__(train_x, train_y, likelihood)
            bs=torch.Size([z_dim])
            self.matern = gpytorch.kernels.MaternKernel(batch_shape=bs)
            self.scale = gpytorch.kernels.ScaleKernel(self.matern,batch_shape=bs)
            self.grid = gpytorch.kernels.GridInterpolationKernel(self.scale, num_dims=train_x.shape[1], grid_size=100)
            self.mean_ = gpytorch.means.ConstantMean(batch_shape=bs)
            self.mean_module = self.mean_
            self.covar_module = self.grid
            self.feature_extractor = FeatureExtractor(train_x.shape[1])
            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            mvn = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(mvn)

        def get_covaraince(self, x, xp):
            proj_x = self.feature_extractor(x)
            proj_x = self.scale_to_bounds(proj_x)
            proj_xp = self.feature_extractor(xp)
            proj_xp = self.scale_to_bounds(proj_xp)            
            cov = self.covar_module(proj_x, proj_xp).to_dense()
            K = cov.mean(axis=0).cpu().numpy().squeeze()

            return K


def train(i_query, z_dim, data, time, np_model, n_iterations):
    # collect current z predictions
    # with torch.no_grad():
    t = torch.from_numpy(time.astype(np.float32))
    t = t.repeat(data.x.shape[0], 1).to(device)
    z, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2),
    data.y.unsqueeze(2))

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=z_dim, rank=0).to(device)
    model = GPModel(data.x, z, likelihood, z_dim).to(device)

    # Setup model training
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters()},
        {'params': model.covar_module.parameters()},
        {'params': model.mean_module.parameters()},
        {'params': model.likelihood.parameters()},
    ], lr=1e-3)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for i in range(n_iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(data.x)
        # Calc loss and backprop gradients
        loss = -mll(output, z)
        loss.backward(retain_graph=True)

        optimizer.step()
        # print('GP Model training : %d - loss : %.3f'%(i, loss.item()))

    return model, loss.item()

data = ActiveLearningDataset(C_initial,y_initial)
gp_model,_ = train(0, N_LATENT, data, time, np_model, N_GP_ITERATIONS)

## Perform active learning campaign
np_model_losses = []
gp_model_losses = []
for i in range(N_QUERIES):
    query_idx = query_strategy(gp_model, C_pool, n_instances=1)
    # query_idx = RNG.choice(range(len(C_pool)),size=1,replace=False)    
    data.update(C_pool[query_idx], y_pool[query_idx])
    colomap_indx.append(i+1)

    if np.remainder(100*(i)/N_QUERIES,10)==0:
        plot_iteration(query_idx, time, data, gp_model, np_model, utility, N_QUERIES, C_train, N_LATENT, colomap_indx) 
        np_model, np_loss = update_npmodel(time, np_model, data)
        plt.savefig(SAVE_DIR+'itr_%d.png'%i)
        np_model_losses.append(np_loss)

    gp_model, gp_loss = train(i, N_LATENT,data, time, np_model, N_GP_ITERATIONS)
    print('Iter %d/%d - Loss: %.3f ' % (i+1, N_QUERIES, gp_loss))
    gp_model_losses.append(gp_loss)

    # remove queried instance from pool
    C_pool = np.delete(C_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

# Plotting after training
plot_loss_profiles(np_model_losses, gp_model_losses, SAVE_DIR+'losses.png')
plot_iteration(query_idx, time, data, gp_model, np_model, utility, N_QUERIES, C_train, N_LATENT, colomap_indx) 
plt.savefig(SAVE_DIR+'itr_%d.png'%i)
plot_phasemap_pred(sim, time, gp_model, np_model, SAVE_DIR)
plot_gpmodel(time, gp_model, np_model, C_train, y_train, SAVE_DIR+'model_c2z.png')    
plot_npmodel(time, N_LATENT, np_model, SAVE_DIR+'samples_in_latentgrid.png')