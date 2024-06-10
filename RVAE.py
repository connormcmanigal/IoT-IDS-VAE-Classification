import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__innit__()
        self.fc1=nn.Linear(input_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.mu=nn.Linear(hidden_dim, latent_dim)
        self.sigma=nn.Linear(hidden_dim, latent_dim)
        self.tanh=nn.Tanh()
    def forward(self, x):
        x=self.tanh(self.fc1(x))
        x=self.tanh(self.fc2())
        mu=self.mu(x)
        sigma=self.sigma(x)
        return mu,sigma
    
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dim):
        super(Encoder,self).__innit__()
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
    # i will finish dis when i wake up
    def __init__(self, latent_dim, in_out_dim, beta, hidden_dim=256):
        super(RVAE,self).__innit__()
        self.encoder=Encoder(in_out_dim, latent_dim, hidden_dim=hidden_dim)
        self.decoder=Decoder(in_out_dim, latent_dim, hidden_dim=hidden_dim)
        self.beta=beta
    def forward(self, x):
        mu, sigma= self.encoder(x)
        z=self.sample(mu,sigma)
        recon=self.decoder(z)
        return recon, mu, sigma

    def sample(self, mu, sigma):
        norm_rand=torch.randint_like(mu)
        z=(norm_rand*mu)+sigma
        return z


def beta_elbow(x_hat, x, beta, N, z_mu, z_sigma):
    # https://arxiv.org/pdf/2006.08204
    # this loss assumes that all columns are continuous which i believe is true
    beta_cross_entropy=-((beta+1)/beta) * torch.mean(torch.exp((-0.5 * z_sigma * beta) * torch.sum((x_hat - x)**2))-1)
    kl_div=torch.mean(0.5 * torch.sum(1 + 2 * torch.log(z_sigma) - (z_mu**2)  - (z_sigma**2), -1))
    return beta_cross_entropy + kl_div
