import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 200, z_dim = 20):
        super(VariationalAutoEncoder, self).__init__()
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
    
    def elbo_loss_function(self, reconstructed_class, true_class, z_mu, z_sigma):
        reconstruction_error = F.binary_cross_entropy(reconstructed_class, true_class, reduction = 'none')
        reconstruction_error = torch.mean(torch.sum(reconstruction_error, -1))
        kl_div = 0.5 * torch.sum(1 + 2 * torch.log(z_sigma) - (z_mu**2) - (z_sigma**2), -1)
        kl_div = torch.mean(kl_div)
        elbo = reconstruction_error - kl_div
        return elbo

    def forward(self, x):
        # return x reconstructed as well as mu and sigma due to use in KL divergence
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decoder(z_reparametrized)
        return x_reconstructed, mu, sigma
     