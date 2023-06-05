import math
import torch
from torch.utils.data import DataLoader, Dataset
import gpytorch
from matplotlib import pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import os, shutil, pdb
SAVE_DIR = './gpytorch_example/'
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)
print('Saving the results to %s'%SAVE_DIR)

BATCH_SIZE = 2
RNG = np.random.default_rng()
N_QUERIES = 10
N_ITERATIONS = 20

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
noise = torch.randn(train_x.size()) * math.sqrt(0.04)
train_y = torch.sin(train_x * (2 * math.pi)) + noise

class ActiveLearningDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = self.to_tensor(x,y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xs = self.x[idx]
        ys = self.y[idx]

        return xs, ys 

    def to_tensor(self, x, y):
        x_ = torch.tensor(x, dtype=torch.float32).to(device)
        y_ = torch.tensor(y, dtype=torch.float32).to(device)

        return x_, y_
    
    def update(self, x, y):
        x, y = self.to_tensor(x, y)
        self.x = torch.hstack((self.x, x))
        self.y = torch.hstack((self.y, y))

        return

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(data):
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(data.x, data.y, likelihood).to(device)

    # Setup model training
    model.train()
    likelihood.train()

    # Use the adam optimizer
    # Includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(N_ITERATIONS):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(data.x.to(device))
        # Calc loss and backprop gradients
        loss = -mll(output, data.y.to(device))
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, N_ITERATIONS, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    return model

def predict(model, x):
    # Test points are regularly spaced along [0,1]
    # Make predictions by querying the mean and covar modules
    model.eval()
    with torch.no_grad():
        dist = model(x.to(device))
        return dist.loc, dist.stddev

def plot(data, model, itr):
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Plot training data as black stars
        ax.plot(data.x.cpu().numpy(), data.y.cpu().numpy(), 'o')
        # Plot predictive means as blue line
        test_x = torch.linspace(0, 1, 51)
        mu, var = predict(model, test_x)
        ax.plot(test_x.numpy(), mu.cpu().numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), 
        (mu-var).cpu().numpy(), (mu+var).cpu().numpy(), alpha=0.5)
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.savefig(SAVE_DIR+'itr_%d.png'%itr)

# Active learning functions
def utility(model, x, beta=5):
    mu, sigma = predict(model, x)

    return (mu + beta*sigma).cpu().numpy()

def selector(values, n_instances):
    max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
    
    return max_idx

def query_strategy(model, x, n_instances=1):
    u = utility(model, x)
    query_idx = selector(u, n_instances)

    return query_idx


# assemble initial data
n_initial = 5
initial_idx = RNG.choice(range(len(train_x)),
                         size=n_initial, 
                         replace=False
                         )
x_initial = train_x[initial_idx,...]
y_initial = train_y[initial_idx,...]

# generate the pool
# remove the initial data from the training dataset
x_pool = np.delete(train_x, initial_idx, axis=0)
y_pool = np.delete(train_y, initial_idx, axis=0)

data = ActiveLearningDataset(x_initial,y_initial)
model = train(data)
plot(data, model, 0)
## Perform active learning campaign
for i in range(N_QUERIES):
    query_idx = query_strategy(model, x_pool, n_instances=1)
    data.update(x_pool[query_idx], y_pool[query_idx])
    model = train(data)
    plot(data, model, i+1)
    # remove queried instance from pool
    x_pool = np.delete(x_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    print(data.x[-1], data.y[-1])