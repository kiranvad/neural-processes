import torch 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
RNG = np.random.default_rng()
import sys, pdb
from activephasemap.activelearn.pipeline import utility, from_comp_to_spectrum
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from scipy import stats
import seaborn as sns 

# plot samples in the composition grid of p(y|c)
def _inset_spectra(c, time, mu, sigma, ax, show_sigma=True):
        loc_ax = ax.transLimits.transform(c)
        ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
        ins_ax.plot(time, mu)
        if show_sigma:
            ins_ax.fill_between(time,mu-sigma, mu+sigma,
            alpha=0.2, color='grey')
        ins_ax.axis('off')
        
        return

def plot_gpmodel_grid(ax, time, C_train, gp_model, np_model, **kwargs):
    c1 = np.linspace(min(C_train[:,0]),max(C_train[:,0]),10)
    c2 = np.linspace(min(C_train[:,1]),max(C_train[:,1]),10)
    with torch.no_grad():
        for i in range(10):
            for j in range(10):
                ci = np.array([c1[i], c2[j]]).reshape(1, 2)
                mu, sigma = from_comp_to_spectrum(time, gp_model, np_model, ci)
                mu_ = mu.cpu().squeeze().numpy()
                sigma_ = sigma.cpu().squeeze().numpy()
                _inset_spectra(ci.squeeze(), time, mu_, sigma_, ax, **kwargs)
    ax.set_xlabel('C1', fontsize=20)
    ax.set_ylabel('C2', fontsize=20)

    return  

def plot_iteration(query_idx, time, data, gp_model, np_model, utility, n_queries, C_train, z_dim, colomap_indx):
    cmap = plt.get_cmap('Reds')
    norm = Normalize(vmin=0, vmax=n_queries)

    layout = [['A1','A2', 'C', 'C'], 
              ['B1', 'B2', 'C', 'C']
              ]
    fig, axs = plt.subplot_mosaic(layout, figsize=(4*4, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    x_ = data.x.cpu().numpy()
    axs['A1'].scatter(x_[:,0], x_[:,1], marker='x', color='k')
    axs['A1'].set_xlabel('C1', fontsize=20)
    axs['A1'].set_ylabel('C2', fontsize=20)    
    axs['A1'].set_title('C sampling')

    axs['A2'].tricontourf(C_train[:,0], C_train[:,1], 
    utility(gp_model, C_train).squeeze(), cmap='plasma')
    axs['A2'].set_title('utility')
    axs['A2'].set_xlabel('C1', fontsize=20)
    axs['A2'].set_ylabel('C2', fontsize=20) 

    with torch.no_grad():
        for _ in range(5):
            c_dim = C_train.shape[1]
            ci = RNG.choice(C_train).reshape(1, c_dim)
            mu, _ = from_comp_to_spectrum(time, gp_model, np_model, ci)
            axs['B2'].plot(time, mu.cpu().squeeze(), color='grey')
            axs['B2'].set_title('random sample p(y|c)')
            axs['B2'].set_xlabel('t', fontsize=20)
            axs['B2'].set_ylabel('f(t)', fontsize=20) 
            z_sample = torch.randn((1, z_dim)).to(device)
            t = torch.from_numpy(time.astype(np.double))
            t = t.view(1, t.shape[0], 1).to(device)
            mu, _ = np_model.xz_to_y(t, z_sample)
            axs['B1'].plot(time, mu.cpu().squeeze(), color='grey')
            axs['B1'].set_title('random sample p(y|z)')
            axs['B1'].set_xlabel('t', fontsize=20)
            axs['B1'].set_ylabel('f(t)', fontsize=20) 

    plot_gpmodel_grid(axs['C'], time, C_train, gp_model, np_model, show_sigma=False)

    return 

def plot_npmodel(time, z_dim, model, fname):
    # plot samples in the latent grid of p(y|z)
    t = torch.from_numpy(time.astype(np.double)).to(device)
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
                mu, sigma = model.xz_to_y(t.to(device), 
                z_sample.to(device))
                mu_ = mu.cpu().squeeze().numpy() 
                sigma_ = sigma.cpu().squeeze().numpy()
                axs[i,9-j].plot(time, mu_)
                # axs[i,9-j].fill_between(time, 
                # mu_-sigma_, mu_+sigma_,alpha=0.2, color='grey')
                axs[i,9-j].set_title('(%.2f, %.2f)'%(z1[i], z2[j]))
                axs[i, 9-j].axis('off')
        fig.supxlabel('z1', fontsize=20)
        fig.supylabel('z2', fontsize=20)

        plt.savefig(fname)
        plt.close()

def plot_gpmodel(time, gp_model, np_model, C_train, y_train, fname):
    # plot comp to z model predictions and the GP covariance
    z_dim = np_model.z_dim
    fig, axs = plt.subplots(4,z_dim, figsize=(4*z_dim, 4*4))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    n_train = len(C_train)
    with torch.no_grad():
        c = torch.tensor(C_train, dtype=torch.double).to(device)
        z_pred = gp_model(c).mean.cpu().numpy()

        t = torch.from_numpy(time.astype(np.double))
        t = t.repeat(n_train, 1).to(device)
        y =  torch.from_numpy(y_train.astype(np.double)).to(device)
        z_true_mu, z_true_sigma = np_model.xy_to_mu_sigma(t.unsqueeze(2),y.unsqueeze(2))
        z_true_mu = z_true_mu.cpu().numpy()
        z_true_sigma = z_true_sigma.cpu().numpy()

        # compare z values from GP and NP models
        for i in range(z_dim):
            sns.kdeplot(z_true_mu[:,i], ax=axs[0,i], fill=True, label='NP Model')
            sns.kdeplot(z_pred[:,i], ax=axs[0,i],fill=True, label='GP Model')
            axs[0,i].set_xlabel('z_%d'%(i+1)) 
            axs[0,i].legend()

        # plot the covariance matrix      
        X,Y = np.meshgrid(np.linspace(min(C_train[:,0]),max(C_train[:,0]),10), 
        np.linspace(min(C_train[:,1]),max(C_train[:,1]),10))
        c_grid_np = np.vstack([X.ravel(), Y.ravel()]).T 
        c_grid = torch.tensor(c_grid_np, dtype=torch.double).to(device)
        # plot covariance of randomly selected points
        idx = RNG.choice(range(n_train),size=z_dim, replace=False)  
        for i, id_ in enumerate(idx):
            ci = C_train[id_,:].reshape(1, 2)
            ci = torch.tensor(ci, dtype=torch.double).to(device)
            Ki = gp_model.get_covaraince(ci, c_grid)
            axs[1,i].tricontourf(c_grid_np[:,0], c_grid_np[:,1], Ki, cmap='plasma')
            axs[1,i].scatter(C_train[id_,0], C_train[id_,1], marker='x', s=50, color='k')
            axs[1,i].set_xlabel('C1')
            axs[1,i].set_ylabel('C2')    

        # plot predicted z values as contour plots
        for i in range(z_dim):
            norm=plt.Normalize(z_pred[:,i].min(),z_pred[:,i].max())
            axs[2,i].tricontourf(C_train[:,0], C_train[:,1], 
            z_pred[:,i], cmap='bwr', norm=norm)        
            axs[2,i].set_xlabel('C1')
            axs[2,i].set_ylabel('C2') 
            axs[2,i].set_title('Predicted z_%d'%(i+1))

        # plot true z values as contour plots
        for i in range(z_dim):
            norm=plt.Normalize(z_true_mu[:,i].min(),z_true_mu[:,i].max())
            axs[3,i].tricontourf(C_train[:,0], C_train[:,1], 
            z_true_mu[:,i], cmap='bwr', norm=norm)        
            axs[3,i].set_xlabel('C1')
            axs[3,i].set_ylabel('C2') 
            axs[3,i].set_title('True z_%d'%(i+1))        

        plt.savefig(fname)
        plt.close()        
    return 

# plot phase map predition
def plot_phasemap_pred(sim, time, gp_model, np_model, fname):
    c_dim = sim.points.shape[1]
    with torch.no_grad():
        idx = RNG.choice(range(len(sim.points)),
                            size=10, 
                            replace=False
                            )
        # plot comparision of predictions with actual
        fig, axs = plt.subplots(2,5, figsize=(4*5, 4*2))
        axs = axs.flatten()
        for i, id_ in enumerate(idx):
            ci = sim.points[id_,:].reshape(1, c_dim)        
            mu, sigma = from_comp_to_spectrum(time, gp_model, np_model, ci)
            mu_ = mu.cpu().squeeze()
            sigma_ = sigma.cpu().squeeze()
            f = sim.F[id_]
            axs[i].scatter(time, f, color='k')
            axs[i].plot(time, mu_, color='k')
            axs[i].fill_between(time,mu_-sigma_, 
            mu_+sigma_,alpha=0.2, color='grey')
        plt.savefig(fname)
        plt.close()

def plot_loss_profiles(np_model_losses, gp_model_losses, fname):
    # plot loss profiles
    fig, axs = plt.subplots(1,2, figsize=(4*2, 4))
    axs[0].plot(np.arange(len(np_model_losses)), np_model_losses, '-o')
    axs[0].set_title('NP Model losses')
    axs[1].plot(np.arange(len(gp_model_losses)), gp_model_losses, '-o')
    axs[1].set_title('GP Model losses')
    plt.savefig(fname)
    plt.close()