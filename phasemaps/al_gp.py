import os, shutil, pdb, sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import torch
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(['./simulators', './plot', './activelearn'])
from simulators import *
from plot import *
from activelearn import *

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
sim = NoisyPhaseSimulator(n_grid=100, use_random_warping=False)
sim.generate()
time = sim.t
sim.plot(SAVE_DIR+'phasemap.png')

# Specify the Neural Process model
np_model = NeuralProcess(1, 1, 50, N_LATENT, 50).to(device)
np_model.load_state_dict(torch.load('./pretrain/trained_model.pt', map_location=device))

C_train, y_train, C_initial, y_initial, C_pool, y_pool, colomap_indx = generate_pool(sim, N_INITIAL,RNG)

data = ActiveLearningDataset(C_initial,y_initial)
gp_model,_ = train(0, N_LATENT, data, time, np_model, N_GP_ITERATIONS)

## Perform active learning campaign
for i in range(N_QUERIES):
    query_idx = query_strategy(gp_model, C_pool, n_instances=1)
    # query_idx = RNG.choice(range(len(C_pool)),size=1,replace=False)    
    data.update(C_pool[query_idx], y_pool[query_idx])
    colomap_indx.append(i+1)
    gp_model, loss = train(i, N_LATENT,data, time, np_model, N_GP_ITERATIONS)
    print('Iter %d/%d - Loss: %.3f ' % (i+1, N_QUERIES, loss))

    if np.remainder(100*(i)/N_QUERIES,10)==0:
        update_npmodel(time, np_model, data)
        plot_iteration(query_idx, time, data, gp_model, np_model, utility, N_QUERIES, C_train, N_LATENT, colomap_indx) 
        plt.savefig(SAVE_DIR+'itr_%d.png'%i)

    # remove queried instance from pool
    C_pool = np.delete(C_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

# freeze model training
np_model.training = False

"""Plotting after training"""
plot_phasemap_pred(sim, time, gp_model, np_model, SAVE_DIR)
plot_gpmodel(time, gp_model, np_model, C_train, y_train, SAVE_DIR+'model_c2z.png')    
plot_npmodel(time, N_LATENT, np_model, SAVE_DIR+'samples_in_latentgrid.png')