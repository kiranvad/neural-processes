import numpy as np 

# create synthetic data
class PhasemapSimulator:
    def __init__(self, n_grid=50, n_domain=100):
        """ Simulate a phasemap with domain warping of functions
        """
        self.n_domain = n_domain
        self.t = np.linspace(0,1, num=self.n_domain)
        x = np.linspace(0,1, n_grid)
        y = np.linspace(0,1, n_grid)
        X,Y = np.meshgrid(x,y)
        self.points = np.vstack([X.ravel(), Y.ravel()]).T
        self.phase1 = lambda x : 0.5*(x)**2+0.45
        self.phase2 = lambda x : -0.45*(x)**2+0.55
        
    def g(self, t, p):
        out = np.ones(self.t.shape)
        for i in range(1,p+1):
            zi = np.random.normal(1, 0.1)
            mean = (2*i-1)/(2*p)
            std = 1/(3*p)
            out += zi*self.phi(t, mean, std)

        return out
    
    def phi(self, t, mu, sigma):
        factor = 1/(2*(sigma**2))
        return np.exp(-factor*(t-mu)**2)
    
    def gamma(self):
        a = np.random.uniform(-3, 3)
        if a==0:
            gam = self.t
        else:
            gam = (np.exp(a*self.t)-1)/(np.exp(a)-1)

        return gam

    def simulate(self, c):
        label = self.get_label(c)
        y = self.g(self.gamma(), label)

        return y
    
    def get_label(self, c):
        if c[1]-self.phase1(c[0])>0:
            label = 1
        elif c[1]-self.phase2(c[0])<0:
            label = 2
        else:
            label = 3
            
        return label

    def generate(self):
        self.labels = [self.get_label(ci) for ci in self.points]            
        self.F = [self.simulate(ci) for ci in self.points]

        return


