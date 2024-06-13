import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self,input_dim, z_dim, h_dim):
        super(Encoder, self).__init__()
        self.fc1=nn.Linear(input_dim, h_dim)
        self.mu=nn.Linear(h_dim, z_dim)
        self.sigma=nn.Linear(h_dim, z_dim)
        self.relu=nn.ReLU()
    def forward(self, x):
        x=self.relu(self.fc1(x))
        mu=self.mu(x)
        sigma=self.sigma(x)
        return mu,sigma
    
class Decoder(nn.Module):
    def __init__(self, input_dim, z_dim, h_dim):
        super(Decoder,self).__init__()
        self.fc1=nn.Linear(z_dim, h_dim)
        self.out=nn.Linear(h_dim, input_dim)
        self.relu=nn.ReLU()
    def forward(self, z):
        z=self.relu(self.fc1(z))
        recon=self.out(z)
        return recon

class VAE4(nn.Module):
    def __init__(self, z_dim, input_dim, h_dim):
        super(VAE4,self).__init__()
        self.encoder=Encoder(input_dim, z_dim, h_dim)
        self.decoder=Decoder(input_dim, z_dim, h_dim)
    def forward(self, x):
        mu, sigma= self.encoder(x)
        z=self.sample(mu,sigma)
        recon=self.decoder(z)
        return recon, mu, sigma

    def sample(self, mu, sigma):
        norm_rand=torch.randn_like(sigma)
        z=mu+(sigma*norm_rand)
        return z

    def loss(self, reconstructed_data, true_data, z_mu, z_sigma, cat_feature_indices):
        """Wrapper function, because proper class inheritance is for nerds"""
        return self.combo_elbo(reconstructed_data, true_data, z_mu, z_sigma, cat_feature_indices)
    
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
        return elbo, recon_tot