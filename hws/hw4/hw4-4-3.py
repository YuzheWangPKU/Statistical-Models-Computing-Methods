import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision

from tqdm import tqdm
#%matplotlib inline


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron
    """
    def __init__(self, input_dim=28*28, output_dim=28*28, n_hidden_layers=1, hidden_dim=512, binary_output=False):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        for i in range(n_hidden_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_dim, output_dim))
        if binary_output:
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
        
class VariationalAutoEncoder(nn.Module):
    """
    Variational Auto-Encoder
    """
    def __init__(self, input_dim, output_dim, latent_dim=2, n_hidden_layers=1, hidden_dim=512):
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = MultiLayerPerceptron(input_dim, 2*latent_dim, n_hidden_layers, hidden_dim)
        self.decoder = MultiLayerPerceptron(latent_dim, output_dim, n_hidden_layers, hidden_dim, binary_output=True)


    def forward(self, x):
        z = self.encoder(x)
        mu, log_sigma = z[:,:self.latent_dim], z[:,self.latent_dim:]

        z = self.reparameterize(mu, log_sigma)
        x_recon = self.decoder(z)

        return x_recon, mu, log_sigma


    def reparameterize(self, mu, log_sigma):
        """
        reparameterization trick \n
        z = \mu + \epsilon * \sigma
        """
        epsilon = torch.randn_like(log_sigma)
        z = mu + epsilon * torch.exp(log_sigma)

        return z


    def loss_function(self, x, x_recon, mu, log_sigma):
        """
        loss function
        https://ai.stackexchange.com/questions/24564/how-does-the-implementation-of-the-vaes-objective-function-equate-to-elbo
        """
        recon_loss = nn.BCELoss(reduction='sum')(x_recon, x)
        kl_div = -0.5 * torch.sum(1 + 2*log_sigma - mu**2 - torch.exp(2*log_sigma))

        return recon_loss + kl_div


    def sample(self, n_samples):
        """
        randomly sample from the latent space
        """
        z = torch.randn(n_samples, self.latent_dim)
        x_recon = self.decoder(z)

        return x_recon


    def to_latent(self, x):
        """
        convert data to latent space
        """
        z = self.encoder(x)
        mu = z[:,:self.latent_dim]

        return mu


    def generate_from_latent(self, z):
        """
        generate data from latent space
        """
        x_recon = self.decoder(z)

        return x_recon


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


def train(model, train_data, test_data, device, n_epochs=100, lr=1e-3, weight_decay=1e-5, save_path=os.getcwd()):
    """
    train VAE model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_list = []
    test_loss_list = []

    for epoch in tqdm(range(n_epochs)):
        train_loss = 0
        for x, _ in train_data:
            x = x.view(-1, 28*28).to(device)

            x_recon, mu, log_sigma = model(x)
            loss = model.loss_function(x, x_recon, mu, log_sigma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            #https://pytorch.org/docs/stable/generated/torch.Tensor.item.html

        train_loss /= len(train_data.dataset)
        train_loss_list.append(train_loss)

        test_loss = 0
        for x, _ in test_data:
            x = x.view(-1, 28*28).to(device)

            x_recon, mu, log_sigma = model(x)
            loss = model.loss_function(x, x_recon, mu, log_sigma)

            test_loss += loss.item()

        test_loss /= len(test_data.dataset)
        test_loss_list.append(test_loss)

    if save_path is not None:
        model.save(save_path)

    return train_loss_list, test_loss_list


MNIST_train_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('MNIST_train_data', train=True, download=True, transform=torchvision.transforms.ToTensor()),
    batch_size=1000, shuffle=True
)
MNIST_test_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('MNIST_test_data', train=False, download=True, transform=torchvision.transforms.ToTensor()),
    batch_size=1000, shuffle=True
)

VAE_model = VariationalAutoEncoder(28*28, 28*28, latent_dim=2, n_hidden_layers=1, hidden_dim=512)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAE_model.to(device)

train_loss_list, test_loss_list = train(VAE_model, MNIST_train_data, MNIST_test_data, device='cuda', n_epochs=100, lr=1e-3, weight_decay=1e-5, save_path='VAE_model')

fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})
ax.plot(np.arange(1, len(train_loss_list)+1, dtype=int), np.array(train_loss_list)*(-1), label='$\\mathcal{L}^{\\rm train}_{1000}$')
ax.plot(np.arange(1, len(test_loss_list)+1, dtype=int), np.array(test_loss_list)*(-1), label='$\\mathcal{L}^{\\rm test}_{1000}$')
ax.set_xlabel('$\\rm epoch$', fontsize='x-large')
ax.set_ylabel('$\\mathcal{L}_{1000}$', fontsize='x-large')


fig.set_size_inches(8,6)
plt.legend(fontsize='x-large')
plt.savefig('4-4-3.jpg',dpi=1000, bbox_inches='tight')
plt.show()


