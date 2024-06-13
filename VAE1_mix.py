import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self,input_dim, latent_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1=nn.Linear(input_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.mu=nn.Linear(hidden_dim, latent_dim)
        self.sigma=nn.Linear(hidden_dim, latent_dim)
        self.relu=nn.ReLU()
    def forward(self, x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        mu=self.mu(x)
        sigma=self.sigma(x)
        return mu,sigma
    
class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hidden_dim):
        super(Decoder,self).__init__()
        self.fc1=nn.Linear(latent_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.out=nn.Linear(hidden_dim, output_dim)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax(dim=1)
    def forward(self, z):
        z=self.relu(self.fc1(z))
        z=self.relu(self.fc2(z))
        recon=self.out(z)
        return recon

class VAE3(nn.Module):
    def __init__(self, latent_dim, in_out_dim, hidden_dim=100):
        super(VAE3,self).__init__()
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
    

    def combo_elbo(self,reconstructed_data, true_data, z_mu, z_sigma, cat_feature_indicies):
        # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
        # Supposedly more robust than MSE.

        # Splitting categorical and continous
        cat_mask=torch.zeros(reconstructed_data.size(1), dtype=torch.bool)
        cat_mask[cat_feature_indicies]=True
        cont_mask=~cat_mask

        recon_cat = reconstructed_data[:, cat_mask]
        recon_cont = reconstructed_data[:, cont_mask]
        true_cat = true_data[:, cat_mask]
        true_cont = true_data[:, cont_mask]

        hubert=F.smooth_l1_loss(recon_cont, true_cont, reduction='none')
        hubert=torch.mean(torch.sum(hubert, dim=-1))
        
        #https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
        bce_logits=F.binary_cross_entropy_with_logits(recon_cat, true_cat, reduction='none')
        bce_logits=torch.mean(torch.sum(bce_logits, dim=-1))

        kl_divergence = -0.5 * torch.sum(1 + 2 * torch.log(z_sigma.clamp(min=1e-8)) - z_mu**2 - z_sigma**2, -1)
        kl_divergence = torch.mean(kl_divergence)

        recon_tot = hubert + bce_logits

        elbo = recon_tot + kl_divergence
        return elbo

    def loss(self, reconstructed_data, true_data, z_mu, z_sigma, cat_feature_indices):
        """Wrapper function, because proper class inheritance is for nerds"""
        return self.combo_elbo(reconstructed_data, true_data, z_mu, z_sigma, cat_feature_indices)

# def beta_elbo(x_hat, x, beta, z_mu, z_sigma, sigma=1):
#     # something is wrong with this i dont know what is going on i give up some numbers are not adding up

#     # https://arxiv.org/pdf/2006.08204
#     # https://arxiv.org/pdf/1905.09961
#     D = x_hat.shape[1]
#     beta_term = (beta+1)/beta
#     mse=torch.pow(x_hat-x, 2)
#     mse=torch.sum(mse, dim=1)
#     mse=torch.mean(mse)
#     constant_term = 1 / (np.power((2 * np.pi * sigma**2),((beta * D)/2)))

#     exponential_term = torch.exp((-beta/(2*sigma**2)) * mse)
#     exponential_mean=torch.mean(exponential_term)

#     loss = beta_term * (constant_term * exponential_mean - 1)
#     # mse=F.mse_loss(x_hat, x, reduction='none')
#     # loss=torch.mean(torch.sum(mse, dim=-1))

#     kl_divergence = -0.5 * torch.sum(1 + 2 * torch.log(z_sigma.clamp(min=1e-8)) - z_mu**2 - z_sigma**2, -1)
#     kl_divergence = torch.mean(kl_divergence)

#     elbow = loss + kl_divergence
#     # print(x_hat)
#     # print(x)
#     print(f'MSE: {loss}')
#     # print(f'Exponential Term: {exponential_mean}, Constant: {constant_term}, KL-Div: {kl_divergence}, MSE:{mse}')
#     return elbow



