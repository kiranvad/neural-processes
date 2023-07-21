import os, shutil, pdb, sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import torch
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from activephasemap.activelearn.simulators import PrabolicPhases, GaussianPhases
from activephasemap.np.neural_process import NeuralProcess
from activephasemap.activelearn import pipeline, visuals, surrogates
from activephasemap.activelearn.pipeline import utility 

SAVE_DIR = './gp/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

BATCH_SIZE = 2
RNG = np.random.default_rng()
N_INITIAL = 5 # number of initial points for AL
N_QUERIES = 100 # Total number of queries to simulator/oracle
N_GP_ITERATIONS = 100 # number of GP fitting iterations
N_LATENT = 3 
N_COMPOSITION = 2

# Define the simulator 
sim = PrabolicPhases(n_grid=100, use_random_warping=False, noise=True)
sim.generate()
time = sim.t
sim.plot(SAVE_DIR+'phasemap.png')

# Specify the Neural Process model
np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load('./pretrain/trained_model.pt', map_location=device))

C_train, y_train, C_initial, y_initial, C_pool, y_pool, colomap_indx = pipeline.generate_pool(sim, N_INITIAL,RNG)

data = pipeline.ActiveLearningDataset(C_initial,y_initial)
gp_model,_ = surrogates.train_gp(0, N_LATENT, data, time, np_model, N_GP_ITERATIONS)

## Perform active learning campaign
np_model_losses = []
gp_model_losses = []
for i in range(N_QUERIES):
    query_idx = pipeline.query_strategy(gp_model, C_pool, n_instances=1)
    # query_idx = RNG.choice(range(len(C_pool)),size=1,replace=False)    
    data.update(C_pool[query_idx], y_pool[query_idx])
    colomap_indx.append(i+1)

    if np.remainder(100*(i)/N_QUERIES,10)==0:
        visuals.plot_iteration(query_idx, time, data, gp_model, np_model, utility, N_QUERIES, C_train, N_LATENT, colomap_indx)
        plt.savefig(SAVE_DIR+'itr_%d.png'%i)
        plt.close()
        visuals.plot_gpmodel(time, gp_model, np_model, C_train, y_train, SAVE_DIR+'gpmodel_itr_%d.png'%i)   
        np_model, np_loss = surrogates.update_npmodel(time, np_model, data)
        np_model_losses.append(np_loss)
    gp_model, gp_loss = surrogates.train_gp(i, N_LATENT,data, time, np_model, N_GP_ITERATIONS)
    print('Iter %d/%d - Loss: %.3f ' % (i+1, N_QUERIES, gp_loss))
    gp_model_losses.append(gp_loss)
    # remove queried instance from pool
    C_pool = np.delete(C_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

"""Plotting after training"""
visuals.plot_loss_profiles(np_model_losses, gp_model_losses, SAVE_DIR+'losses.png')
visuals.plot_iteration(query_idx, time, data, gp_model, np_model, utility, N_QUERIES, C_train, N_LATENT, colomap_indx)
plt.savefig(SAVE_DIR+'itr_%d.png'%i) 
visuals.plot_phasemap_pred(sim, time, gp_model, np_model, SAVE_DIR+'compare_spectra_pred.png')
visuals.plot_gpmodel(time, gp_model, np_model, C_train, y_train, SAVE_DIR+'model_c2z.png')    
visuals.plot_npmodel(time, N_LATENT, np_model, SAVE_DIR+'samples_in_latentgrid.png')