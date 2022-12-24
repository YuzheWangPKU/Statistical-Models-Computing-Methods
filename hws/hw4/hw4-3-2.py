import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from tqdm import tqdm
#%matplotlib inline

data = np.load('banana_shape_data.npy')


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, n_hidden_layers=2, input_dim=1, hidden_dim=32, output_dim=1):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for i in range(n_hidden_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
        
class PlanarFlow(nn.Module):
    """
    Implementation of Planar Flows
    """
    def __init__(self, shape=2):
        super(PlanarFlow, self).__init__()
        self.w = nn.Parameter(torch.randn(shape)[None])
        self.u = nn.Parameter(torch.randn(shape)[None])
        self.b = nn.Parameter(torch.randn(1))
        
    
    def forward(self, z):
        uTw = torch.sum(self.w * self.u)
        u_hat = self.u + (torch.log(1 + torch.exp(uTw)) - 1 - uTw) * self.w / torch.sum(self.w**2)
        
        wTz = torch.sum(self.w * z, list(range(1, self.w.dim())), keepdim=True)
        # https://medium.com/analytics-vidhya/an-intuitive-understanding-on-tensor-sum-dimension-with-pytorch-d9b0b6ebbae

        log_det = torch.log(torch.abs(1 + torch.sum(u_hat * self.w) / (torch.cosh(wTz + self.b).reshape(-1))**2))
        z = z + u_hat * torch.tanh(wTz + self.b)

        return z, log_det
    

class NICE(nn.Module):
    """
    Implementation of NICE (Nonlinear Independent Components Estimation)
    """
    def __init__(self, shape=2):
        super(NICE, self).__init__()
        self.s = nn.Parameter(torch.randn(shape)[None])
        self.mlp1, self.mlp2 = [MultiLayerPerceptron(n_hidden_layers=2, input_dim=shape//2, hidden_dim=32, output_dim=shape//2) for i in range(2)]
            
    def forward(self, z):
        z1, z2 = z.chunk(2, dim=1)
        z2 = z2 + self.mlp1(z1)
        z1 = z1 + self.mlp2(z2)
        z = torch.cat([z1, z2], dim=1)
        z = z * torch.abs(self.s)

        log_det = torch.sum(torch.log(torch.abs(self.s)))
        
        return z, log_det


class RealNVP(nn.Module):
    """
    Implementation of RealNVP (Real-valued Non-Volume Preserving)
    """
    def __init__(self, shape=2):
        super(RealNVP, self).__init__()
        self.mlp1, self.mlp2, self.mlp3, self.mlp4 = [MultiLayerPerceptron(n_hidden_layers=2, input_dim=shape//2, hidden_dim=32, output_dim=shape//2) for i in range(4)]

    def forward(self, z):
        z1, z2 = z.chunk(2, dim=1)
        log_det_term1 = self.mlp1(z1)
        z2 = z2 * torch.exp(log_det_term1) + self.mlp2(z1)
        log_det_term2 = self.mlp3(z2)
        z1 = z1 * torch.exp(log_det_term2) + self.mlp4(z2)
        z = torch.cat([z1, z2], dim=1)

        log_det = (log_det_term1 + log_det_term2).reshape(-1)

        return z, log_det


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model
    """
    def __init__(
        self,
        model_type,
        train_data=data,
        shape=2,
        n_flow=4,
        sample_size=100
    ):
        super(NormalizingFlow, self).__init__()
        self.train_data = train_data
        self.shape = shape
        self.sample_size = sample_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if model_type == 'Planar':
            self.flows = nn.ModuleList([PlanarFlow(shape) for i in range(n_flow)])
        elif model_type == 'NICE':
            self.flows = nn.ModuleList([NICE(shape) for i in range(n_flow)])
        elif model_type == 'RealNVP':
            self.flows = nn.ModuleList([RealNVP(shape) for i in range(n_flow)])
        else:
            raise NotImplementedError('This type of normalizing flow model is yet to be implemented.')


    def compute_ELBO(self):
        """
        compute ELBO estimated via Monte Carlo
        """
        sample = torch.tensor(np.random.normal(size=(self.sample_size, self.shape)), dtype=torch.float32, device=self.device)
        sample_init = sample.detach().clone()
        train_data = torch.tensor(self.train_data, dtype=torch.float32, device=self.device)

        log_det_sum = torch.zeros(self.sample_size, dtype=torch.float32, device=self.device)
        for flow in self.flows:
            sample, log_det = flow(sample)
            log_det_sum += log_det

        sample_theta1, sample_theta2 = sample.chunk(2, dim=1)

        ELBO_term1 = torch.sum(torch.log(torch.exp(-0.125 * (sample_theta1 + sample_theta2**2 - train_data)**2) / (2*np.sqrt(2*torch.pi))), dim=1)
        ELBO_term2 = -0.5 * torch.sum(sample**2, dim=1) - np.log(2*np.pi)
        ELBO_term3 = 0.5 * torch.sum(sample_init**2, dim=1) + np.log(2*np.pi)
        ELBO_term4 = log_det_sum

        computed_ELBO = torch.mean(ELBO_term1 + ELBO_term2 + ELBO_term3 + ELBO_term4)

        return computed_ELBO


    def draw_sample(self, n_sample=500):
        """
        draw samples of amount `n_sample` from the model
        """
        sample = torch.tensor(np.random.normal(size=(n_sample, self.shape)), dtype=torch.float32, device=self.device)
        for flow in self.flows:
            sample, _ = flow(sample)

        return sample.detach().clone().to('cpu').numpy()


    def save(self, path):
        """
        save state dict of the model
        """
        torch.save(self.state_dict(), path)


    def load(self, path):
        """
        load model from state dict
        """
        self.load_state_dict(torch.load(path))


# Planar Flows 
model = NormalizingFlow(model_type='Planar', n_flow=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_iter = 2000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_list = []

for iter in tqdm(range(num_iter)):
    optimizer.zero_grad()
    loss = -model.compute_ELBO()
    if not (torch.isnan(loss) or torch.isinf(loss)):
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().clone().to('cpu').numpy())
    else:
        loss_list.append(loss_list[-1])

model.save('Planar_model')
learned_sample = model.draw_sample(n_sample=500)

fig, (ax1, ax2) = plt.subplots(1,2)
plt.rcParams.update({
    "text.usetex": True
})
ax1.plot(np.arange(1, num_iter+1, dtype=int), np.array(loss_list)*(-1), linewidth=1)
ax1.set_xlabel('$\\rm epoch$', fontsize='x-large')
ax1.set_ylabel('$\\rm ELBO$', fontsize='x-large')

ax2.scatter(learned_sample[:,0], learned_sample[:,1], s=5)
ax2.set_xlabel('$\\theta_1$', fontsize='x-large')
ax2.set_ylabel('$\\theta_2$', fontsize='x-large')

fig.set_size_inches(16,6)
plt.savefig('4-3-2-1.jpg',dpi=1000, bbox_inches='tight')
plt.show()


# NICE
model = NormalizingFlow(model_type='NICE', n_flow=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_iter = 10000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_list = []

for iter in tqdm(range(num_iter)):
    optimizer.zero_grad()
    loss = -model.compute_ELBO()
    if not (torch.isnan(loss) or torch.isinf(loss)):
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().clone().to('cpu').numpy())
    else:
        loss_list.append(loss_list[-1])

model.save('NICE_model')
learned_sample = model.draw_sample(n_sample=500)

fig, (ax1, ax2) = plt.subplots(1,2)
plt.rcParams.update({
    "text.usetex": True
})
ax1.plot(np.arange(1, num_iter+1, dtype=int), np.array(loss_list)*(-1), linewidth=1)
ax1.set_xlabel('$\\rm epoch$', fontsize='x-large')
ax1.set_ylabel('$\\rm ELBO$', fontsize='x-large')

ax2.scatter(learned_sample[:,0], learned_sample[:,1], s=5)
ax2.set_xlabel('$\\theta_1$', fontsize='x-large')
ax2.set_ylabel('$\\theta_2$', fontsize='x-large')

fig.set_size_inches(16,6)
plt.savefig('4-3-2-2.jpg',dpi=1000, bbox_inches='tight')
plt.show()


# RealNVP
model = NormalizingFlow(model_type='RealNVP', n_flow=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_iter = 10000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_list = []

for iter in tqdm(range(num_iter)):
    optimizer.zero_grad()
    loss = -model.compute_ELBO()
    if not (torch.isnan(loss) or torch.isinf(loss)):
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().clone().to('cpu').numpy())
    else:
        loss_list.append(loss_list[-1])

model.save('RealNVP_model')
learned_sample = model.draw_sample(n_sample=500)

fig, (ax1, ax2) = plt.subplots(1,2)
plt.rcParams.update({
    "text.usetex": True
})
ax1.plot(np.arange(1, num_iter+1, dtype=int), np.array(loss_list)*(-1), linewidth=1)
ax1.set_xlabel('$\\rm epoch$', fontsize='x-large')
ax1.set_ylabel('$\\rm ELBO$', fontsize='x-large')

ax2.scatter(learned_sample[:,0], learned_sample[:,1], s=5)
ax2.set_xlabel('$\\theta_1$', fontsize='x-large')
ax2.set_ylabel('$\\theta_2$', fontsize='x-large')

fig.set_size_inches(16,6)
plt.savefig('4-3-2-3.jpg',dpi=1000, bbox_inches='tight')
plt.show()

