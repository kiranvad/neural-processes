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

sys.path.append('/mmfs1/home/kiranvad/kiranvad/hyak/SAXSpy/neural-processes/')
from neural_process import NeuralProcess
from training import NeuralProcessTrainer
from utils import context_target_split

sys.path.append('./helpers.py')
from helpers import PhasemapSimulator, ActiveLearningDataset

SAVE_DIR = './active_learn_gp/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

BATCH_SIZE = 2
RNG = np.random.default_rng()
N_INITIAL = 25 # number of initial points for AL
N_QUERIES = 10 # Total number of queries to simulator/oracle
N_GP_ITERATIONS = 65 # number of GP fitting iterations
N_LATENT = 3 
N_COMPOSITION = 2

# Define the simulator 
sim = PhasemapSimulator(n_grid=100, use_random_warping=True)
sim.generate()
time = sim.t

fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
axs = axs.T
c = np.linspace(0, 1, 10)
for i in range(10):
    for j in range(10):
        cij = np.array([c[i], c[j]])
        axs[i,9-j].plot(time, sim.simulate(cij))
        axs[i,9-j].set_xlim(0, 1)
        axs[i, 9-j].axis('off')
fig.supxlabel('C1', fontsize=20)
fig.supylabel('C2', fontsize=20) 
plt.savefig(SAVE_DIR+'phasemap.png')
plt.close()

# Specify the Neural Process model
x_dim = 1
y_dim = 1
r_dim = 50  
z_dim = N_LATENT  
h_dim = 50 

np_model = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim).to(device)
np_model.load_state_dict(torch.load('./pretrain/trained_model.pt',
map_location=device))

class NPModelDataset(Dataset):
    def __init__(self, y):
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

def update_npmodel(np_model, data):
    batch_size = 2
    num_context = 75
    num_target = 25
    dataset = NPModelDataset(data.y)
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

    return

# Define a GP (batch independent)
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([N_LATENT]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([N_LATENT])),
            batch_shape=torch.Size([N_LATENT])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


def train(i_query, data, np_model):
    # collect current z predictions
    with torch.no_grad():
        t = torch.from_numpy(time.astype(np.float32))
        t = t.repeat(data.x.shape[0], 1).to(device)
        z, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2),
        data.y.unsqueeze(2))

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=N_LATENT, rank=0).to(device)
    model = GPModel(data.x, z, likelihood).to(device)

    # Setup model training
    model.train()
    likelihood.train()

    # Use the adam optimizer
    # Includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(N_GP_ITERATIONS):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(data.x)
        # Calc loss and backprop gradients
        loss = -mll(output, z)
        loss.backward()

        optimizer.step()
        # print('%d - loss : %.3f'%(i, loss.item()))

    print('Iter %d/%d - Loss: %.3f ' % (i_query + 1, N_QUERIES, loss.item()))

    return model

def predict(model, x):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32).to(device)
        dist = model(x)

        return dist.loc.cpu().numpy(), dist.stddev.cpu().numpy()

def from_comp_to_spectrum(gp_model, np_model, c):
    with torch.no_grad():
        c = torch.tensor(c, dtype=torch.float32).to(device)
        z_sample,_ = predict(gp_model, c)
        z_sample = torch.tensor(z_sample, dtype=torch.float32).to(device)
        t = torch.from_numpy(time.astype(np.float32)).to(device)
        t = t.repeat(c.shape[0]).view(c.shape[0], len(time), 1)
        mu, std = np_model.xz_to_y(t, z_sample)

        return mu, std    

# Active learning functions
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

# assemble initial data
C_train = sim.points.astype(np.float32)
y_train = np.asarray(sim.F, dtype=np.float32)

# assemble initial data
initial_idx = RNG.choice(range(len(C_train)),
                         size=N_INITIAL, 
                         replace=False
                         )
C_initial = C_train[initial_idx,:]
y_initial = y_train[initial_idx,:]
colomap_indx = [0]*N_INITIAL
# generate the pool
# remove the initial data from the training dataset
C_pool = np.delete(C_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)

data = ActiveLearningDataset(C_initial,y_initial)
gp_model = train(0, data, np_model)

cmap = plt.get_cmap('Reds')
norm = Normalize(vmin=0, vmax=N_QUERIES)

def plot_iteration(query_idx, gp_model, np_model):
    fig, axs = plt.subplots(1,4, figsize=(4*4, 4))
    fig.suptitle('Query %d'%i)
    x_ = data.x.cpu().numpy()
    axs[0].scatter(x_[:,0], x_[:,1], 
            c=colomap_indx, cmap=cmap, norm=norm)
    axs[0].set_title('C sampling')

    with torch.no_grad():
        for _ in range(5):
            ci = RNG.choice(C_pool).reshape(1, N_COMPOSITION)
            mu, _ = from_comp_to_spectrum(gp_model, np_model, ci)
            axs[1].plot(time, mu.cpu().squeeze())
            axs[1].set_title('random sample p(y|c)')

            z_sample = torch.randn((1, z_dim)).to(device)
            t = torch.from_numpy(time.astype(np.float32))
            t = t.view(1, t.shape[0], 1).to(device)
            mu, _ = np_model.xz_to_y(t, z_sample)
            axs[2].plot(time, mu.cpu().squeeze())
            axs[2].set_title('random sample p(y|z)')

    axs[3].tricontourf(C_train[:,0], C_train[:,1], 
    utility(gp_model, C_train).squeeze(), cmap='plasma')
    axs[3].set_title('utility')

    return 

## Perform active learning campaign
for i in range(N_QUERIES):
    # query_idx = query_strategy(gp_model, C_pool, n_instances=1)
    query_idx = RNG.choice(range(len(C_pool)),size=1,replace=False)    
    data.update(C_pool[query_idx], y_pool[query_idx])
    colomap_indx.append(i+1)
    gp_model = train(i, data, np_model)

    if np.remainder(100*(i)/N_QUERIES,10)==0:
        update_npmodel(np_model, data)
        plot_iteration(query_idx, gp_model, np_model) 
        plt.savefig(SAVE_DIR+'itr_%d.png'%i)

    # remove queried instance from pool
    C_pool = np.delete(C_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

# freeze model training
np_model.training = False

"""Plotting after training"""

# plot phase map predition
with torch.no_grad():
    fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
    cmap = plt.get_cmap('Spectral')
    norm = Normalize(vmin=0, vmax=3)
    axs[0].scatter(sim.points[:,0], sim.points[:,1], 
                   c=sim.labels, alpha=0.2, cmap=cmap, norm=norm
                   )
    idx = RNG.choice(range(len(sim.points)),
                         size=10, 
                         replace=False
                         )

    # plot prediction of spectra given composition
    for i in idx:
        ci = sim.points[i,:].reshape(1, N_COMPOSITION)
        mu, _ = from_comp_to_spectrum(gp_model, np_model, ci)
        mu_ = mu.cpu().squeeze()
        axs[1].plot(time, mu_, color=cmap(norm(sim.labels[i]))) 
        axs[0].scatter(sim.points[i,0], sim.points[i,1], 
        color='k', marker='x')
    plt.savefig(SAVE_DIR+'final_sample.png')
    plt.close()

    # plot comparision of predictions with actual
    fig, axs = plt.subplots(1,5, figsize=(4*5, 4))
    for i in range(5):
        ci = sim.points[idx[i],:].reshape(1, N_COMPOSITION)        
        mu, sigma = from_comp_to_spectrum(gp_model, np_model, ci)
        mu_ = mu.cpu().squeeze()
        sigma_ = sigma.cpu().squeeze()
        f = sim.F[idx[i]]
        axs[i].plot(time, f, color='k')
        axs[i].plot(time, mu_, ls='--', color='k')
        axs[i].fill_between(time,mu_-sigma_, 
        mu_+sigma_,alpha=0.2, color='grey')
    plt.savefig(SAVE_DIR+'final_compare.png')
    plt.close()

# plot samples in the composition grid of p(y|c)
c1 = torch.linspace(0,1,10)
c2 = torch.linspace(0,1,10)
fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
axs = axs.T
with torch.no_grad():
    for i in range(10):
        for j in range(10):
            ci = np.array([c1[i], c2[j]]).reshape(1, N_COMPOSITION)
            mu, sigma = from_comp_to_spectrum(gp_model, np_model, ci)
            mu_ = mu.cpu().squeeze().numpy()
            sigma_ = sigma.cpu().squeeze().numpy()
            axs[i,9-j].plot(time, mu_)
            axs[i,9-j].fill_between(time,mu_-sigma_, mu_+sigma_,
            alpha=0.2, color='grey')
            axs[i,9-j].set_xlim(0, 1)
            axs[i,9-j].set_title('(%.2f, %.2f)'%(c1[i], c2[j]))
            axs[i, 9-j].axis('off')
    fig.supxlabel('C1', fontsize=20)
    fig.supylabel('C2', fontsize=20) 

    plt.savefig(SAVE_DIR+'samples_in_compgrid.png')
    plt.close()

# plot samples in the latent grid of p(y|z)
t = torch.from_numpy(time.astype(np.float32)).to(device)
t = t.unsqueeze(1).unsqueeze(0)

z1 = torch.linspace(-3,3,10)
z2 = torch.linspace(-3,3,10)
fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
axs = axs.T
with torch.no_grad():
    for i in range(10):
        for j in range(10):
            z_sample = torch.zeros((1, z_dim))
            z_sample[0,0] = z1[i]
            z_sample[0,1] = z2[j]
            mu, sigma = np_model.xz_to_y(t.to(device), 
            z_sample.to(device))
            mu_ = mu.cpu().squeeze().numpy() 
            sigma_ = sigma.cpu().squeeze().numpy()
            axs[i,9-j].plot(time, mu_)
            axs[i,9-j].fill_between(time, 
            mu_-sigma_, mu_+sigma_,alpha=0.2, color='grey')
            axs[i,9-j].set_xlim(0, 1)
            axs[i,9-j].set_title('(%.2f, %.2f)'%(z1[i], z2[j]))
            axs[i, 9-j].axis('off')
    fig.supxlabel('z1', fontsize=20)
    fig.supylabel('z2', fontsize=20)

    plt.savefig(SAVE_DIR+'samples_in_latentgrid.png')
    plt.close()


# plot comp to z model predictions and the GP covariance
fig, axs = plt.subplots(1,4, figsize=(4*4, 4))
n_train = len(C_train)
with torch.no_grad():
    c = torch.tensor(C_train, dtype=torch.float32).to(device)
    dist = gp_model(c)
    z_pred = dist.loc.cpu().numpy()
    z_pred = z_pred.reshape(n_train,3)

    t = torch.from_numpy(time.astype(np.float32))
    t = t.repeat(n_train, 1).to(device)
    y =  torch.from_numpy(y_train.astype(np.float32)).to(device)
    z_true, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2),
    y.unsqueeze(2))
    z_true = z_true.cpu().numpy()
    for i in range(3):
        axs[i].scatter(z_true[:,i], z_pred[:,i], color='k')
        axs[i].plot(np.sort(z_true[:,i]), np.sort(z_true[:,i]), 
        color='k', ls='--')
        axs[i].set_title('z_%d'%(i+1))
        axs[i].set_xlim([z_true[:,i].min(), z_true[:,i].max()])
        axs[i].set_ylim([z_true[:,i].min(), z_true[:,i].max()]) 

    # plot the covariance matrix      
    X,Y = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))
    c_grid_np = np.vstack([X.ravel(), Y.ravel()]).T 
    c_grid = torch.tensor(c_grid_np, dtype=torch.float32).to(device)
    K = gp_model.covar_module(c_grid).to_dense()
    K = K.mean(axis=0).cpu().numpy()
    energy = K.mean(axis=1)
    axs[3].tricontourf(c_grid_np[:,0], c_grid_np[:,1], energy, 
    cmap='plasma')
    plt.savefig(SAVE_DIR+'model_c2z.png')




