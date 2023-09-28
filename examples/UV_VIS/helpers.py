import glob
import numpy as np
import torch
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
RNG = np.random.default_rng()
import pdb
import glob
from scipy.spatial.distance import cdist
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import interpolate

class UVVisDataset(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the images.
        """
        self.dir = root_dir
        self.files = glob.glob(self.dir+'/*.npz')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            npzfile = np.load(self.files[i])
        except Exception as e:
            print('%s Could not load %s'%(type(e).__name__, self.files[i]))
        wl, I = npzfile['wl'], npzfile['I']
        wl = (wl-min(wl))/(max(wl)-min(wl))
        wl_ = torch.tensor(wl.astype(np.float32)).unsqueeze(1)
        I_ = torch.tensor(I.astype(np.float32)).unsqueeze(1)

        return wl_, I_

class GNPPhases:
    def __init__(self):
        comps = pd.read_csv('./gold_nano_grid/grid.csv').to_numpy()
        files = glob.glob('./gold_nano_grid/Grid_*.xlsx')
        self.spectra = [pd.read_excel(file) for file in files]
        AG_x = self.minmax(comps[:,0]*0.00064/350*10**5)
        AA_x = self.minmax(comps[:,1]*0.00630/350*10**4)
        self.points = np.hstack((AG_x.reshape(-1,1), AA_x.reshape(-1,1)))
        self.wl = self.spectra[0]['Wavelength'].values.astype('double')
        wl_ = np.linspace(min(self.wl), max(self.wl), num=100)
        self.t = (wl_-min(wl_))/(max(wl_)-min(wl_))
        
    def simulate(self, c):
        rid = np.random.choice(len(self.spectra))
        lookup_dist = cdist(c.reshape(1,-1), self.points)
        lookup_cid = np.argmin(lookup_dist)
        y = self.spectra[rid].iloc[:,lookup_cid+1].values.astype('double')
        wl = self.spectra[rid]['Wavelength'].values.astype('double')
        spline = interpolate.splrep(wl, y, s=0)
        wl_ = np.linspace(min(wl), max(wl), num=100)
        I_grid = interpolate.splev(wl_, spline, der=0)
        norm = np.sqrt(np.trapz(I_grid**2, wl_))

        return I_grid/norm 
    
    def generate(self):
        self.F = [self.simulate(ci) for ci in self.points]

        return

    def minmax(self, c):
        return (c-min(c))/(max(c)-min(c))

    def plot(self, fname=None):
        fig, axs = plt.subplots(10,10, figsize=(2*10, 2*10))
        axs = axs.T
        c1 = np.linspace(min(self.points[:,0]), max(self.points[:,0]), 10)
        c2 = np.linspace(min(self.points[:,1]), max(self.points[:,1]), 10)
        for i in range(10):
            for j in range(10):
                cij = np.array([c1[i], c2[j]])
                axs[i,9-j].plot(self.t, self.simulate(cij))
        fig.supxlabel('C1', fontsize=20)
        fig.supylabel('C2', fontsize=20) 
        if fname is not None:
            plt.savefig(fname)
            plt.close()
        else:
            plt.show()