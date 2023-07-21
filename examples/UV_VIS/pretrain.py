import sys, os, pdb, shutil
sys.path.append('/mmfs1/home/kiranvad/kiranvad/neural-processes')
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import MultiPeakGaussians
from math import pi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from neural_process import NeuralProcess
from torch.utils.data import DataLoader
from training import NeuralProcessTrainer
from utils import context_target_split

sys.path.append('./helpers.py')
from helpers import UVVisDataset

SAVE_DIR = './results_pretrain/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
os.makedirs(SAVE_DIR+'/itrs/')
print('Saving the results to %s'%SAVE_DIR)

batch_size = 2
num_context = 25
num_target = 25
num_iterations = 30
x_dim = 1
y_dim = 1
r_dim = 50  # Dimension of representation of context points
z_dim = 2  # Dimension of sampled latent variable
h_dim = 50  # Dimension of hidden layers in encoder and decoder
learning_rate = 1e-3

# Create dataset
dataset = UVVisDataset(root_dir='./uvvis_data_npy')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
x, y = next(iter(data_loader))
print(x.shape, y.shape)
# Visualize data samples
for i in np.random.randint(len(dataset), size=100):
    xi, yi = dataset[i]
    plt.plot(xi.cpu().numpy(), yi.cpu().numpy(), c='b', alpha=0.5)
plt.savefig(SAVE_DIR+'data_samples.png')
plt.close()

neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim)
# Create a set of 100 target points, with shape 
# (batch_size, num_points, x_dim), which in this case is
# (1, 100, 1)
x_target = torch.Tensor(np.linspace(0, 1, 100))
x_target = x_target.unsqueeze(1).unsqueeze(0)

for i in range(100):
    z_sample = torch.randn((1, z_dim))
    mu, _ = neuralprocess.xz_to_y(x_target, z_sample)
    plt.plot(x_target.numpy()[0], mu.detach().numpy()[0], c='b', alpha=0.5)
plt.savefig(SAVE_DIR+'samples_before_training.png')

optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=learning_rate)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(num_context, num_context),
                                  num_extra_target_range=(num_target, num_target), 
                                  print_freq=1000)

neuralprocess.training = True
x_plot = torch.Tensor(np.linspace(0, 1, 100))
x_plot = x_plot.unsqueeze(1).unsqueeze(0)
np_trainer.train(data_loader, num_iterations, x_plot, savedir=SAVE_DIR+'/itrs/') 

neuralprocess.training = False

with torch.no_grad():
    for i in range(100):
        z_sample = torch.randn((1, z_dim))
        mu, _ = neuralprocess.xz_to_y(x_target, z_sample)
        plt.plot(x_target.numpy()[0], mu.detach().numpy()[0], 
                c='b', alpha=0.5)

    plt.savefig(SAVE_DIR+'samples_after_training.png')
    plt.close()

    # Extract a batch from data_loader
    x, y = next(iter(data_loader))
    x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 
                                                    num_context, 
                                                    num_target)

    for i in range(200):
        # Neural process returns distribution over y_target
        p_y_pred = neuralprocess(x_context, y_context, x_target)
        # Extract mean of distribution
        mu = p_y_pred.loc.detach()
        plt.plot(x_target.numpy()[0], mu.numpy()[0], alpha=0.05, c='b')

    plt.scatter(x_context[0].numpy(), y_context[0].numpy(), c='k')
    plt.savefig(SAVE_DIR+'samples_from_posterior.png')
    plt.close()

    # plot grid
    z1 = torch.linspace(-3,3,10)
    z2 = torch.linspace(-3,3,10)
    fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
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
            # axs[i,j].set_title('(%.2f, %.2f)'%(z1[i], z2[j]))
    fig.supxlabel('z1')
    fig.supylabel('z2')

    plt.savefig(SAVE_DIR+'samples_in_grid.png')
    plt.close()

# %%
torch.save(neuralprocess.state_dict(), SAVE_DIR+'trained_model.pt')