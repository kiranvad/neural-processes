import os, shutil, pdb, sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import torch
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append('/mmfs1/home/kiranvad/kiranvad/neural-processes')
sys.path.append('./helpers')
from neural_process import NeuralProcess
from helpers import GNPPhases
from activelearn_plot import *
from activelearn import *

SAVE_DIR = './gp/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

BATCH_SIZE = 2
RNG = np.random.default_rng()
N_INITIAL = 5 # number of initial points for AL
N_QUERIES = 10 # Total number of queries to simulator/oracle
N_GP_ITERATIONS = 100 # number of GP fitting iterations
N_LATENT = 2
N_COMPOSITION = 2

# Define the simulator 
sim = GNPPhases()
sim.generate()
time = sim.t
sim.plot(fname=SAVE_DIR+'phasemap.png')

# Specify the Neural Process model
np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load('./results_pretrain/trained_model.pt', map_location=device))

C_train, y_train, C_initial, y_initial, C_pool, y_pool, colomap_indx = generate_pool(sim, N_INITIAL,RNG)

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
        plt.savefig(SAVE_DIR+'itr_%d.png'%i)
        plt.close()
        plot_gpmodel(time, gp_model, np_model, C_train, y_train, SAVE_DIR+'gpmodel_itr_%d.png'%i)   
        np_model, np_loss = update_npmodel(time, np_model, data)
        np_model_losses.append(np_loss)
    gp_model, gp_loss = train(i, N_LATENT,data, time, np_model, N_GP_ITERATIONS)
    print('Iter %d/%d - Loss: %.3f ' % (i+1, N_QUERIES, gp_loss))
    gp_model_losses.append(gp_loss)
    # remove queried instance from pool
    C_pool = np.delete(C_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

"""Plotting after training"""
plot_loss_profiles(np_model_losses, gp_model_losses, SAVE_DIR+'losses.png')

plot_iteration(query_idx, time, data, gp_model, np_model, utility, N_QUERIES, C_train, N_LATENT, colomap_indx)
plt.savefig(SAVE_DIR+'itr_%d.png'%i)

plot_phasemap_pred(sim, time, gp_model, np_model, SAVE_DIR+'final_compare.png')
plot_gpmodel(time, gp_model, np_model, C_train, y_train, SAVE_DIR+'model_c2z.png')    
plot_npmodel(time, N_LATENT, np_model, SAVE_DIR+'samples_in_latentgrid.png')

fig, ax = plt.subplots(figsize=(10,10))
plot_gpmodel_grid(ax, time, C_train, gp_model, np_model, show_sigma=False)
plt.savefig(SAVE_DIR+'phasemap_pred.png') 

torch.save(np_model.state_dict(), SAVE_DIR+'np_model.pt')
torch.save(gp_model.state_dict(), SAVE_DIR+'gp_model.pt')