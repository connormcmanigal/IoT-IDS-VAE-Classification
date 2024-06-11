import torch
import torch.nn as nn
import math


class Encoder(nn.Module):
    def __init__(self,input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1=nn.Linear(input_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.mu=nn.Linear(hidden_dim, latent_dim)
        self.sigma=nn.Linear(hidden_dim, latent_dim)
        self.tanh=nn.Tanh()
    def forward(self, x):
        x=self.tanh(self.fc1(x))
        x=self.tanh(self.fc2(x))
        mu=self.mu(x)
        sigma=self.sigma(x)
        return mu,sigma
    
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dim):
        super(Decoder,self).__init__()
        self.fc1=nn.Linear(latent_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.out=nn.Linear(hidden_dim, output_dim)
        self.tanh=nn.Tanh()
    def forward(self, z):
        z=self.tanh(self.fc1(z))
        z=self.tanh(self.fc2(z))
        recon=self.out(z)
        return recon

class RVAE(nn.Module):
    def __init__(self, latent_dim, in_out_dim, hidden_dim=4400):
        super(RVAE,self).__init__()
        self.encoder=Encoder(in_out_dim, latent_dim, hidden_dim=hidden_dim)
        self.decoder=Decoder(in_out_dim, latent_dim, hidden_dim=hidden_dim)
    def forward(self, x):
        mu, sigma= self.encoder(x)
        z=self.sample(mu,sigma)
        recon=self.decoder(z)
        return recon, mu, sigma

    def sample(self, mu, sigma):
        norm_rand=torch.randn_like(sigma)
        z=mu+(sigma*norm_rand)
        return z


def beta_elbow(x_hat, x, beta, z_mu, z_sigma):
    # https://arxiv.org/pdf/2006.08204
    # this loss assumes that all columns are continuous which i believe is true
    # beta_cross_entropy=-((beta+1)/beta) * torch.mean(torch.exp((-0.5 * z_sigma * beta) * torch.sum((x_hat - x)**2, dim=-1))-1)
    beta_cross_entropy=-((beta+1)/beta) * torch.mean(((1/(2*math.pi*z_sigma**2)**(beta/2)) * torch.exp((-0.5*beta*z_sigma**2)*torch.sum((x_hat-x)**2, dim=1))-1))
    kl_div=torch.mean(0.5 * torch.sum(1 + 2 * torch.log(z_sigma) - (z_mu**2)  - (z_sigma**2), dim=-1))
    return beta_cross_entropy + kl_div

