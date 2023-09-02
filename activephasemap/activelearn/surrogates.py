import os, shutil, pdb, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal as mvn
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood
from activephasemap.np.neural_process import NeuralProcess
from activephasemap.np.training import NeuralProcessTrainer
from activephasemap.np.utils import context_target_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train_gp(i_query, n_tasks, data, time, np_model, n_iterations):
    # collect current z predictions
    # with torch.no_grad():
    t = torch.from_numpy(time.astype(torch.double))
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  

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
        # print('%d - loss : %.3f'%(i, loss.item()))

    return model, loss.item() 


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

class MinFeatureExtractor(torch.nn.Sequential):
    def __init__(self, dim):
        super().__init__()
        self.add_module('linear1', torch.nn.Linear(dim, 8))
        self.add_module('relu1', torch.nn.ReLU())              
        self.add_module('linear4', torch.nn.Linear(8, 2))

class DKLGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, z_dim):
            super().__init__(train_x, train_y, likelihood)
            bs=torch.Size([z_dim])
            self.matern = gpytorch.kernels.MaternKernel(batch_shape=bs)
            self.scale = gpytorch.kernels.ScaleKernel(self.matern,batch_shape=bs)
            self.grid = gpytorch.kernels.GridInterpolationKernel(self.scale, num_dims=train_x.shape[1], grid_size=100)
            self.mean_ = gpytorch.means.ConstantMean(batch_shape=bs)
            self.mean_module = self.mean_
            self.covar_module = self.grid
            # self.feature_extractor = FeatureExtractor(train_x.shape[1])
            self.feature_extractor = MinFeatureExtractor(train_x.shape[1])
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


def train_dkl(i_query, z_dim, data, time, np_model, n_iterations):
    # collect current z predictions
    # with torch.no_grad():
    t = torch.from_numpy(time.astype(torch.double))
    t = t.repeat(data.x.shape[0], 1).to(device)
    z, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2),
    data.y.unsqueeze(2))

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=z_dim, rank=0).to(device)
    model = DKLGPModel(data.x, z, likelihood, z_dim).to(device)

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


# Neural Process model functions
class NPModelDataset(Dataset):
    def __init__(self, time, y):
        self.data = []
        for yi in y:
            xi = torch.from_numpy(time.astype(torch.double))
            xi = xi.view(xi.shape[0],1).to(device)
            yi = yi.view(yi.shape[0],1).to(device)
            self.data.append((xi,yi))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def update_npmodel(time, np_model, data, **kwargs):
    batch_size = kwargs.get('batch_size',  2)
    num_context = kwargs.get('num_context',  25)
    num_target = kwargs.get('num_target',  25)
    num_iterations = kwargs.get('num_iterations',  30)
    print(data.y.shape)
    dataset = NPModelDataset(time, data.y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    np_optimizer = torch.optim.Adam(np_model.parameters(), lr=1e-3)
    np_trainer = NeuralProcessTrainer(device, 
    np_model, np_optimizer,
    num_context_range=(num_context, num_context),
    num_extra_target_range=(num_target, num_target),
    verbose = False
    )

    np_model.training = True
    np_trainer.train(data_loader, num_iterations)
    loss = np_trainer.epoch_loss_history[-1]
    print('NP model loss : %.2f'%loss)

    # freeze model training
    np_model.training = False

    return np_model, loss 

