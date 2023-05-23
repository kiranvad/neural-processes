import sys, os, pdb, shutil

sys.path.append('/mmfs1/home/kiranvad/kiranvad/hyak/SAXSpy/neural-processes/')
from neural_process import NeuralProcess
from training import NeuralProcessTrainer
from utils import context_target_split

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as get_cmap
from matplotlib.colors import Normalize

import torch
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn, optim
sys.path.append('./helpers.py')
from helpers import PhasemapSimulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = './active_learn/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

# %%
# hyper-parameters
N_TIME = 100
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
RNG = np.random.default_rng()
N_LATENT = 3
N_COMPOSITION = 2
n_initial = 25
n_queries = 100

# Define the simulator
sim = PhasemapSimulator()
sim.generate()
time = sim.t

# Specify the model
x_dim = 1
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 3  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder

np_model = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim).to(device)
np_model.load_state_dict(torch.load('./pretrain/trained_model.pt',
map_location=device))

class ActiveLearningDataset(Dataset):
    def __init__(self, C, y):
        self.C = C
        self.y = y

    def __len__(self):
        return len(self.C)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Cs = torch.tensor(self.C[idx], dtype=torch.float32).to(device)
        ys = torch.tensor(self.y[idx], dtype=torch.float32).to(device)

        return Cs, ys 
    
    def update(self, C, y):
        self.C = np.vstack((self.C, C))
        self.y = np.vstack((self.y, y))
        
        return

class ActiveLearningModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.c2z = nn.Sequential(
            nn.Linear(self.dim,32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            )
        
        self.mu = nn.Linear(64, N_LATENT)

    def forward(self, c):
        h = self.c2z(c)
        mu = self.mu(h)

        return mu

def loss_function(c, mu, model):
    mu_hat = model(c)
    loss = nn.functional.mse_loss(mu, mu_hat, reduction='sum')

    return loss

def train(model, train_loader):
    model.train()
    train_loss = 0
    for ci, yi in train_loader:
        ci = ci.to(device)
        yi = yi.to(device)
        xi = torch.from_numpy(time.astype(np.float32))
        xi = xi.repeat(ci.shape[0], 1).to(device)
        with torch.no_grad():
            mu, _ = np_model.xy_to_mu_sigma(xi.unsqueeze(2), yi.unsqueeze(2))
        optimizer.zero_grad()
        loss = loss_function(ci, mu, model)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss / len(train_loader.dataset)

class NPModelDataset(Dataset):
    def __init__(self, y):
        self.data = []
        for yi in y:
            xi = torch.from_numpy(time.astype(np.float32))
            xi = xi.view(xi.shape[0],1).to(device)
            yi = torch.from_numpy(yi.astype(np.float32))
            yi = yi.view(yi.shape[0],1).to(device)
            self.data.append((xi,yi))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def fit(np_model, model, data):
    batch_size = 2
    num_context = 75
    num_target = 25
    dataset = NPModelDataset(data.y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(np_model.parameters(), lr=3e-4)
    np_trainer = NeuralProcessTrainer(device, np_model, optimizer,
                                    num_context_range=(num_context, num_context),
                                    num_extra_target_range=(num_target, num_target), 
                                    print_freq=1000,
                                    verbose = False
                                    )

    np_model.training = True
    np_trainer.train(data_loader, 30)

    loader = torch.utils.data.DataLoader(data,
    batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    for e in range(NUM_EPOCHS):
        loss = train(model, loader)

    print('p(z|c) loss value : %.4f'%loss)

    return

def predict(model, c):
    c = torch.tensor(c, dtype=torch.float32).to(device)
    with torch.no_grad():
        h = model.c2z(c)
        z_sample = model.mu(h)
        x = torch.from_numpy(time.astype(np.float32)).to(device)
        x = x.repeat(c.shape[0]).view(c.shape[0], len(time), 1)
        mu, std = np_model.xz_to_y(x, z_sample)

    return mu, std

model = ActiveLearningModel(N_COMPOSITION).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %%
C_train = sim.points.astype(np.float32)
y_train = np.asarray(sim.F, dtype=np.float32)

# assemble initial data
initial_idx = RNG.choice(range(len(C_train)),
                         size=n_initial, 
                         replace=False
                         )
C_initial = C_train[initial_idx,:]
y_initial = y_train[initial_idx,:]
colomap_indx = [0]*n_initial
# generate the pool
# remove the initial data from the training dataset
C_pool = np.delete(C_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)

data = ActiveLearningDataset(C_initial,y_initial)

# %%
def utility(model, C_query):
    with torch.no_grad():
        mu, sigma = predict(model, C_query)

    return sigma.mean(axis=1).cpu()

def selector(values, n_instances):
    max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
    return max_idx

def query_strategy(model, C_query, n_instances=1):
    u = utility(model, C_query)
    query_idx = selector(u, n_instances)

    return query_idx

# the active learning loop
cmap = get_cmap['Reds']
norm = Normalize(vmin=0, vmax=n_queries)

def plot_iteration(query_idx, model, np_model):
    fig, axs = plt.subplots(1,4, figsize=(4*4, 4))
    fig.suptitle('Query %d'%i)
    axs[0].scatter(data.C[:,0], data.C[:,1], 
            c=colomap_indx, cmap=cmap, norm=norm)
    axs[0].scatter(C_pool[query_idx,0], C_pool[query_idx,1], 
                color='k', marker='x')
    with torch.no_grad():
        for _ in range(5):
            ci = RNG.choice(C_pool)
            mu, _ = predict(model, ci.reshape(1, N_COMPOSITION))
            axs[1].plot(time, mu.cpu().squeeze())
            axs[1].set_title('p(y|c)')

            z_sample = torch.randn((1, z_dim)).to(device)
            x = torch.from_numpy(time.astype(np.float32))
            x = x.view(1, x.shape[0], 1).to(device)
            mu, _ = np_model.xz_to_y(x, z_sample)
            axs[2].plot(time, mu.cpu().squeeze())
            axs[2].set_title('p(y|z)')

    axs[3].tricontourf(C_train[:,0], C_train[:,1], 
    utility(model, C_train).squeeze(), cmap='plasma')

    return 

for i in range(n_queries):
    # query_idx = query_strategy(model, C_pool, n_instances=1)
    # Check with random selection
    query_idx = RNG.choice(range(len(C_pool)),
                            size=1, 
                            replace=False
                            )
    data.update(C_pool[query_idx], y_pool[query_idx])
    colomap_indx.append(i+1)

    if np.remainder(100*i/n_queries,10)==0:
        fit(np_model, model, data)
        plot_iteration(query_idx, model, np_model) 
        plt.savefig(SAVE_DIR+'itr_%d.png'%i)

    # remove queried instance from pool
    C_pool = np.delete(C_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

# %%
with torch.no_grad():
    fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
    cmap = get_cmap['Spectral']
    norm = Normalize(vmin=0, vmax=3)
    axs[0].scatter(sim.points[:,0], sim.points[:,1], 
                   c=sim.labels, alpha=0.2, cmap=cmap, norm=norm
                   )
    idx = RNG.choice(range(len(sim.points)),
                         size=10, 
                         replace=False
                         )
    mu,std = predict(model, sim.points[idx,:])
    for i, s in zip(idx,mu.cpu()):
        axs[1].plot(time, s.squeeze(), 
        color=cmap(norm(sim.labels[i]))) 
        axs[0].scatter(sim.points[i,0], sim.points[i,1], 
        color='k', marker='x')
    plt.savefig(SAVE_DIR+'final_sample.png')
    plt.close()

    fig, axs = plt.subplots(1,5, figsize=(4*5, 4))
    for i in range(5):
        mu_i = mu[i].cpu().squeeze()
        sigma_i = std[i].cpu().squeeze()
        f = sim.F[idx[i]]
        axs[i].plot(time, f, color='k')
        axs[i].plot(time, mu_i, ls='--', color='k')
        axs[i].fill_between(time,mu_i-sigma_i, 
        mu_i+sigma_i,alpha=0.2, color='grey')
    plt.savefig(SAVE_DIR+'final_compare.png')
    plt.close()