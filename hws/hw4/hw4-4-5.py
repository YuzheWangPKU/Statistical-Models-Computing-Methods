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


def visualize_latent(model, data, device, n_samples=1000):
    """
    visualize latent space
    """
    x, y = next(iter(data))
    x = x.view(-1, 28*28).to(device)

    with torch.no_grad():
        z = model.to_latent(x)

    z = z.cpu().numpy()
    y = y.numpy()

    plt.figure(figsize=(8,6))
    plt.rcParams.update({
        "text.usetex": True
    })
    plt.scatter(z[:n_samples,0], z[:n_samples,1], c=y, cmap='tab10', s=10)
    plt.xlabel('$z_1$', fontsize='x-large')
    plt.ylabel('$z_2$', fontsize='x-large')
    plt.colorbar()
    plt.savefig('4-4-5-1.jpg',dpi=1000, bbox_inches='tight')
    plt.show()


def visualize_generation(model, device, z1_min, z1_max, z2_min, z2_max, n=20):
    """
    Generate and visualize digits from the trained VAE model for each of these grid points
    """
    z1 = np.linspace(z1_min, z1_max, n)
    z2 = np.linspace(z2_max, z2_min, n)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z = np.concatenate((Z1.reshape(-1,1), Z2.reshape(-1,1)), axis=1)
    Z = torch.tensor(Z).float().to(device)

    with torch.no_grad():
        x_recon = model.generate_from_latent(Z)

    x_recon = x_recon.cpu().numpy()

    plt.figure(figsize=(20,20))
    plt.rcParams.update({
        "text.usetex": True
    })
    for i in range(n**2):
        plt.subplot(20,20,i+1)
        plt.imshow(x_recon[i,:].reshape(28,28), cmap='gray')
        plt.axis('off')
    
    plt.savefig('4-4-5-2.jpg',dpi=1000, bbox_inches='tight')
    plt.show()


MNIST_test_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('MNIST_test_data', train=False, download=True, transform=torchvision.transforms.ToTensor()),
    batch_size=1000, shuffle=True
)

VAE_model = VariationalAutoEncoder(28*28, 28*28, latent_dim=2, n_hidden_layers=1, hidden_dim=512)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VAE_model.to(device)

VAE_model.load('VAE_model')
VAE_model.eval()

# generation
visualize_generation(VAE_model, device, z1_min=-3, z1_max=3, z2_min=-3, z2_max=3, n=20)


# latent space visualization
visualize_latent(VAE_model, MNIST_test_data, device, n_samples=1000)
