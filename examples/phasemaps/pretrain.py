import sys, os, pdb, shutil
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import torch
from torch.utils.data import DataLoader

from activephasemap.np.neural_process import NeuralProcess
from activephasemap.np.training import NeuralProcessTrainer
from activephasemap.np.utils import context_target_split
from activephasemap.np.datasets import MultiPeakGaussians

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = './pretrain/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

# Create dataset
dataset = MultiPeakGaussians(num_samples=5000)

# Visualize data samples
for i in range(10):
    x, y = dataset[i] 
    plt.plot(x, y, c='b', alpha=0.5)
    plt.xlim(0, 1)
plt.savefig(SAVE_DIR+'data_samples.png')
plt.close()


x_dim = 1
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 3  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder

neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
# Create a set of 100 target points, with shape 
# (batch_size, num_points, x_dim), which in this case is
# (1, 100, 1)
x_target = torch.Tensor(np.linspace(0, 1, 100))
x_target = x_target.unsqueeze(1).unsqueeze(0)

for i in range(100):
    z_sample = torch.randn((1, z_dim))  # Shape (batch_size, z_dim)
    # Map x_target and z to p_y_target (which is parameterized by a 
    # normal with mean mu and std dev sigma)
    mu, _ = neuralprocess.xz_to_y(x_target, z_sample)
    # Plot predicted mean at each target point (note we could also
    # sample from distribution but plot mean for simplicity)
    plt.plot(x_target.numpy()[0], mu.detach().numpy()[0], 
             c='b', alpha=0.5)
    plt.xlim(0, 1)
plt.savefig(SAVE_DIR+'samples_before_training.png')

batch_size = 2
num_context = 75
num_target = 25

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=1e-3)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(num_context, num_context),
                                  num_extra_target_range=(num_target, num_target), 
                                  print_freq=1000)

neuralprocess.training = True
np_trainer.train(data_loader, 30)    

for i in range(100):
    z_sample = torch.randn((1, z_dim))
    mu, _ = neuralprocess.xz_to_y(x_target, z_sample)
    plt.plot(x_target.numpy()[0], mu.detach().numpy()[0], 
             c='b', alpha=0.5)
    plt.xlim(0, 1)

plt.savefig(SAVE_DIR+'samples_after_training.png')
plt.close()


# Extract a batch from data_loader
for batch in data_loader:
    break

# Use batch to create random set of context points
x, y = batch
x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 
                                                  num_context, 
                                                  num_target)


neuralprocess.training = False

for i in range(200):
    # Neural process returns distribution over y_target
    p_y_pred = neuralprocess(x_context, y_context, x_target)
    # Extract mean of distribution
    mu = p_y_pred.loc.detach()
    plt.plot(x_target.numpy()[0], mu.numpy()[0], 
             alpha=0.05, c='b')

plt.scatter(x_context[0].numpy(), y_context[0].numpy(), c='k')
plt.savefig(SAVE_DIR+'samples_from_posterior.png')
plt.close()

# plot grid
z1 = torch.linspace(-3,3,10)
z2 = torch.linspace(-3,3,10)
fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
with torch.no_grad():
    for i in range(10):
        for j in range(10):
            z_sample = torch.zeros((1, z_dim))
            z_sample[0,0] = z1[i]
            z_sample[0,1] = z2[j]
            mu, sigma = neuralprocess.xz_to_y(x_target, z_sample)
            mu_, sigma_ = mu.squeeze().numpy(), sigma.squeeze().numpy()
            axs[i,j].plot(x_target.squeeze().numpy(), mu_)
            axs[i,j].fill_between(x_target.squeeze().numpy(), 
            mu_-sigma_, mu_+sigma_,alpha=0.2, color='grey')
            axs[i,j].set_xlim(0, 1)
            # axs[i,j].set_title('(%.2f, %.2f)'%(z1[i], z2[j]))
    fig.supxlabel('z1')
    fig.supylabel('z2')

    plt.savefig(SAVE_DIR+'samples_in_grid.png')
    plt.close()

# %%
torch.save(neuralprocess.state_dict(), SAVE_DIR+'trained_model.pt')