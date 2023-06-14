import torch 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
RNG = np.random.default_rng()
import sys, pdb
sys.path.append('./activelearn')
from activelearn import utility, from_comp_to_spectrum
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# plot samples in the composition grid of p(y|c)
def _inset_spectra(c, time, mu, sigma, ax, **kwargs):
        loc_ax = ax.transLimits.transform(c)
        ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
        ins_ax.plot(time, mu)
        ins_ax.fill_between(time,mu-sigma, mu+sigma,
        alpha=0.2, color='grey')
        ins_ax.axis('off')
        
        return

def _plot_gpmodel_grid(ax, time, gp_model, np_model):
    c1 = torch.linspace(0,1,10)
    c2 = torch.linspace(0,1,10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    with torch.no_grad():
        for i in range(10):
            for j in range(10):
                ci = np.array([c1[i], c2[j]]).reshape(1, 2)
                mu, sigma = from_comp_to_spectrum(time, gp_model, np_model, ci)
                mu_ = mu.cpu().squeeze().numpy()
                sigma_ = sigma.cpu().squeeze().numpy()
                _inset_spectra(ci.squeeze(), time, mu_, sigma_, ax)
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
    axs['A1'].scatter(x_[:,0], x_[:,1], 
            c=colomap_indx, cmap=cmap, norm=norm)
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
            t = torch.from_numpy(time.astype(np.float32))
            t = t.view(1, t.shape[0], 1).to(device)
            mu, _ = np_model.xz_to_y(t, z_sample)
            axs['B1'].plot(time, mu.cpu().squeeze(), color='grey')
            axs['B1'].set_title('random sample p(y|z)')
            axs['B1'].set_xlabel('t', fontsize=20)
            axs['B1'].set_ylabel('f(t)', fontsize=20) 

    _plot_gpmodel_grid(axs['C'], time, gp_model, np_model)

    return 

def plot_npmodel(time, z_dim, model, fname):
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
                mu, sigma = model.xz_to_y(t.to(device), 
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

        plt.savefig(fname)
        plt.close()

def plot_gpmodel(time, gp_model, np_model, C_train, y_train, fname):
    # plot comp to z model predictions and the GP covariance
    fig, axs = plt.subplots(4,4, figsize=(4*4, 4*4))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    n_train = len(C_train)
    with torch.no_grad():
        c = torch.tensor(C_train, dtype=torch.float32).to(device)
        z_pred = gp_model(c).mean.cpu().numpy()

        t = torch.from_numpy(time.astype(np.float32))
        t = t.repeat(n_train, 1).to(device)
        y =  torch.from_numpy(y_train.astype(np.float32)).to(device)
        z_true_mu, z_true_sigma = np_model.xy_to_mu_sigma(t.unsqueeze(2),y.unsqueeze(2))
        z_true_mu = z_true_mu.cpu().numpy()
        z_true_sigma = z_true_sigma.cpu().numpy()

        # compare z values from GP and NP models
        for i in range(3):
            axs[0,i].scatter(z_true_mu[:,i], z_pred[:,i], color='k')
            sortidx = np.argsort(z_true_mu[:,i])
            axs[0,i].plot(z_true_mu[sortidx,i], z_true_mu[sortidx,i], color='k', ls='--')
            axs[0,i].fill_between(z_true_mu[sortidx,i], 
            z_true_mu[sortidx,i]+z_true_sigma[sortidx,i], 
            z_true_mu[sortidx,i]-z_true_sigma[sortidx,i],            
            color='k', alpha=0.2)            
            axs[0,i].set_title('z_%d'%(i+1))
            axs[0,i].set_xlim([z_true_mu[:,i].min(), z_true_mu[:,i].max()])
            axs[0,i].set_ylim([z_true_mu[:,i].min(), z_true_mu[:,i].max()]) 
            axs[0,i].set_xlabel('NP Model')
            axs[0,i].set_ylabel('GP Model') 

        # plot the covariance matrix      
        X,Y = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))
        c_grid_np = np.vstack([X.ravel(), Y.ravel()]).T 
        c_grid = torch.tensor(c_grid_np, dtype=torch.float32).to(device)
        K = gp_model.covar_module(c_grid).to_dense()
        K = K.mean(axis=0).cpu().numpy()
        energy = K.mean(axis=1)
        axs[0,3].tricontourf(c_grid_np[:,0], c_grid_np[:,1], energy, 
        cmap='plasma')
        axs[0,3].set_xlabel('C1')
        axs[0,3].set_ylabel('C2')

        # plot covariance of randomly selected points
        idx = RNG.choice(range(n_train),size=4, replace=False)  
        for i, id_ in enumerate(idx):
            ci = C_train[id_,:].reshape(1, 2)
            ci = torch.tensor(ci, dtype=torch.float32).to(device)
            cov = gp_model.covar_module(ci, c_grid).to_dense()
            Ki = cov.mean(axis=0).cpu().numpy().squeeze()
            axs[1,i].tricontourf(c_grid_np[:,0], c_grid_np[:,1], Ki, cmap='plasma')
            axs[1,i].scatter(C_train[id_,0], C_train[id_,1], marker='x', s=50, color='k')
            axs[1,i].set_xlabel('C1')
            axs[1,i].set_ylabel('C2')    

        # plot predicted z values as contour plots
        for i in range(3):
            norm=plt.Normalize(z_pred[:,i].min(),z_pred[:,i].max())
            axs[2,i].tricontourf(C_train[:,0], C_train[:,1], 
            z_pred[:,i], cmap='bwr', norm=norm)        
            axs[2,i].set_xlabel('C1')
            axs[2,i].set_ylabel('C2') 
            axs[2,i].set_title('Predicted z_%d'%(i+1))

        # plot true z values as contour plots
        for i in range(3):
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
def plot_phasemap_pred(sim, time, gp_model, np_model, SAVE_DIR):
    c_dim = sim.points.shape[1]
    with torch.no_grad():
        idx = RNG.choice(range(len(sim.points)),
                            size=5, 
                            replace=False
                            )
        # plot comparision of predictions with actual
        fig, axs = plt.subplots(1,5, figsize=(4*5, 4))
        for i, id_ in enumerate(idx):
            ci = sim.points[id_,:].reshape(1, c_dim)        
            mu, sigma = from_comp_to_spectrum(time, gp_model, np_model, ci)
            mu_ = mu.cpu().squeeze()
            sigma_ = sigma.cpu().squeeze()
            f = sim.F[id_]
            axs[i].plot(time, f, color='k')
            axs[i].plot(time, mu_, ls='--', color='k')
            axs[i].fill_between(time,mu_-sigma_, 
            mu_+sigma_,alpha=0.2, color='grey')
        plt.savefig(SAVE_DIR+'final_compare.png')
        plt.close()