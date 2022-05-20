# coding=utf-8

"""
Definition of VAE using PyTorch Lightning
"""

import copy
from typing import List
import torch
import torch.nn as nn
import medvae.models.encoder as e
import pytorch_lightning as pl

class VAE(pl.LightningModule):
    """
    Implements an Variational Auto-Encoder (VAE)
    architecture. 
    """
    def __init__(self,
                 layer_sizes:List[int],
                 input_size:List[int],
                 latent_dim:int):
        """
        :param layer_sizes: List[int], shape of each FCN.
        :param input_size: List[int], input shape to network.
        """
        super(VAE, self).__init__()

        self.encoder = e.VariationalEncoder(layer_sizes,
                                            input_size,
                                            latent_dim)
        reverse_layer_sizes = copy.deepcopy(layer_sizes)
        reverse_layer_sizes.reverse()
        self.decoder = e.Decoder([latent_dim]+reverse_layer_sizes,
                                 input_size)

    def training_step(self, batch, batch_idx):
        """
        Custom training step for VAE
        """
        x, _ = batch
        p_x = self.encoder(x) # Q(z|x)
        mu, sigma =\
            self.encoder.latent_dim_mu(p_x), self.encoder.latent_dim_sigma(p_x)
        std = torch.exp(sigma / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        x_hat = self.decoder(z) # P(x|z)
        recon_loss = self.encoder.gaussian_likelihood(x_hat,
                                                      self.encoder.log_scale,
                                                      x)
        kl = self.encoder.kl_divergence(z, mu, std)
        loss = (kl - recon_loss)
        return loss

    def configure_optimizers(self):
        """
        """
        raise NotImplementedError()