# coding=utf-8

"""
Definition of CVAE using PyTorch Lightning
"""


from audioop import reverse
import copy
from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import medvae.models.encoder as e
import medvae.models.utils as u
import pytorch_lightning as pl


class CVAEBase(pl.LightningModule):
    """
    Implements a Conditional Variational Auto-Encoder
    (CVAE)
    Base architecture with common procedures
    for network.
    """
    def __init__(self,
                 **kwargs):
        """
        
        """
        super(CVAEBase, self).__init__(**kwargs)
        self.encoder = None
        self.decoder = None
        self.convolutional_encoder = False

    def training_step(self, batch, batch_idx):
        """
        Custom training step for a CVAE.
        We concat the labels to the latent
        """
        x, c = batch
        if not self.convolutional_encoder:
            x_in = torch.flatten(x, start_dim=1)
            x_in = torch.concat((x_in,c), dim=-1)
        else:
            x_in = x
        # Concat labels to data
        p_x = self.encoder(x_in) # Q(z|x, c)
        mu, sigma =\
            self.encoder.latent_dim_mu(p_x), self.encoder.latent_dim_sigma(p_x)
        std = torch.exp(sigma / 2)
        q = torch.distributions.Normal(mu, std)
        z_latent = q.rsample()

        z_latent_c = torch.cat((z_latent, c), dim=-1)
        if self.convolutional_encoder:
            z = self.decoder_bridge(z_latent_c)
            z = self.decoder_bridge_1(z)
        else:
            z = z_latent_c

        x_hat = self.decoder(z.view(-1, *self.encoder.out_shape)) # P(x|z, c)
        if self.convolutional_encoder:
            x_hat = F.relu(self.last_conv(x_hat))
        recon_loss = self.encoder.gaussian_likelihood(x_hat,
                                                      self.encoder.log_scale,
                                                      x)
        kl = self.encoder.kl_divergence(z_latent, mu, std)
        loss = (kl - recon_loss)
        return loss

    def configure_optimizers(self):
        """
        """
        raise NotImplementedError()


class FCCVAE(CVAEBase):
    """
    Implements an Fully Connected
    Conditional Variational Auto-Encoder
    (CVAE) variant architecture. 
    """
    def __init__(self,
                 num_filters:List[int],
                 input_shape:List[int],
                 latent_dim:int,
                 num_labels:int,
                 **kwargs):
        """
        :param layer_sizes: List[int], shape of each FCN.
        :param input_size: List[int], input shape to network.
        :param latent_dim: int, latent layer size.
        :param num_labels: int, number of potential classes.
        """
        super(FCCVAE, self).__init__(**kwargs)
        self.build_network(num_filters, input_shape, latent_dim,num_labels, **kwargs)

    def build_network(self,
                      num_filters,
                      input_shape,
                      latent_dim,
                      num_labels,
                      **kwargs):
        """
        Builds the networks for the FC CVAE.
        :param layer_sizes: List[int], shape of each FCN.
        :param input_size: List[int], input shape to network.
        :param latent_dim: int, latent layer size.
        :param num_labels: int, number of potential classes.
        """
        reverse_layer_sizes = copy.deepcopy(num_filters)
        reverse_layer_sizes.reverse()
        if num_filters[0] != np.prod(input_shape)+num_labels:
            # the initial number of filters does not account for
            # the concatenation of the labels to the input
            # so add this as an extra FCN at the beginning of the
            # network.
            num_filters = [np.prod(input_shape)+num_labels] + num_filters
        if reverse_layer_sizes[0] != latent_dim+num_labels:
            # First FCN for the decoder does not account
            # for the concatenation of the labels to the latent
            # layer output. add it as an extra FCN at the
            # beginning of the reverse_layers network.
            reverse_layer_sizes = [latent_dim+num_labels] + reverse_layer_sizes

        self.encoder = e.FCVariationalEncoder(num_filters,
                                              input_shape,
                                              latent_dim,
                                              **kwargs)
        self.encoder.out_shape = [latent_dim+num_labels]
        self.decoder = e.Decoder(reverse_layer_sizes,
                                 input_shape)
