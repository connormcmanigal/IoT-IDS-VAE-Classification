import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 200, z_dim = 20):
        super().__init__()
        # encoder
        self.input2hidden = nn.Linear(input_dim, h_dim)
        self.hidden2mu = nn.Linear(h_dim, z_dim)
        self.hidden2sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z2hidden = nn.Linear(z_dim, h_dim)
        self.hidden2output = nn.Linear(h_dim, input_dim)

    def encoder(self, x):
        # q_phi(z|x)
        h = F.relu(self.input2hidden(x))
        mu, sigma = self.hidden2mu(h), self.hidden2sigma(h)
        return mu, sigma

    def decoder(self, z):
        # p_theta(x|z)
        h = F.relu(self.z2hidden(z))
        reconstruction = torch.sigmoid(self.hidden2output(h))
        return reconstruction

    def forward(self, x):
        # return x reconstructed as well as mu and sigma due to use in KL divergence
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)
        z_reparam = mu + sigma * epsilon
        x_recon = self.decoder(z_reparam)
        return x_recon, mu, sigma