# coding=utf-8

"""
Definition of VAE using PyTorch Lightning
"""

import copy
from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import medvae.models.encoder as e
import medvae.models.utils as u
import pytorch_lightning as pl

class VAEBase(pl.LightningModule):
    """
    Implements an Variational Auto-Encoder (VAE)
    Base architecture with common procedures
    for network.
    """
    def __init__(self,
                 **kwargs):
        """
        
        """
        super(VAEBase, self).__init__(**kwargs)
        self.encoder = None
        self.decoder = None

    def build_network(self,):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """
        Custom training step for any VAE
        """
        x, _ = batch
        p_x = self.encoder(x) # Q(z|x)
        mu, sigma =\
            self.encoder.latent_dim_mu(p_x), self.encoder.latent_dim_sigma(p_x)
        std = torch.exp(sigma / 2)
        q = torch.distributions.Normal(mu, std)
        z_latent = q.rsample()
        z = self.decoder_bridge(z_latent)
        z = self.decoder_bridge_1(z)

        x_hat = self.decoder(z.view(-1, *self.encoder.out_shape)) # P(x|z)
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

class FCVAE(VAEBase):
    """
    Implements an Fully Connected
    Variational Auto-Encoder (VAE)
    variant architecture. 
    """
    def __init__(self,
                 num_filters:List[int],
                 input_shape:List[int],
                 latent_dim:int,
                 **kwargs):
        """
        :param layer_sizes: List[int], shape of each FCN.
        :param input_size: List[int], input shape to network.
        :param latent_dim: int, latent layer size.
        """
        super(FCVAE, self).__init__(**kwargs)
        self.build_network(num_filters, input_shape, latent_dim, **kwargs)

    def build_network(self,
                      num_filters,
                      input_shape,
                      latent_dim,
                      **kwargs):
        reverse_layer_sizes = copy.deepcopy(num_filters)
        reverse_layer_sizes.reverse()
        self.encoder = e.FCVariationalEncoder(num_filters,
                                              input_shape,
                                              latent_dim,
                                              **kwargs)
        self.decoder = e.Decoder([latent_dim]+reverse_layer_sizes,
                                 input_shape)

class ConvVAE(VAEBase):
    """
    Implements an Convolutional
    Variational Auto-Encoder (VAE)
    variant architecture. 
    """
    def __init__(self,
                 num_filters:List[int],
                 input_shape:List[int],
                 latent_dim:int,
                 kernel_size:List[int],
                 stride:List[int],
                 padding:List[int],
                 padding_mode:List[str],
                 dilation:Union[List[list], List[int]],
                 **kwargs):
        """
        :param layer_sizes: List[int], shape of each FCN.
        :param input_size: List[int], input shape to network.
        :param latent_dim: int, latent layer size.
        """
        super(ConvVAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.build_network(num_filters,
                           input_shape,
                           latent_dim,
                           kernel_size,
                           stride,
                           padding,
                           padding_mode,
                           dilation,
                           **kwargs)

    def build_network(self,
                      num_filters,
                      input_shape,
                      latent_dim,
                      kernel_size:List[int],
                      stride:List[int],
                      padding:List[int],
                      padding_mode:List[str],
                      dilation:Union[List[int], List[list]],
                      **kwargs):
        """
        Builds convolutional VAE architecture.

        """
        self.num_filters = num_filters
        reverse_layer_sizes = copy.deepcopy(num_filters)
        reverse_layer_sizes.reverse()
        self.encoder = e.ConvVariationalEncoder(num_filters,
                                                input_shape,
                                                latent_dim,
                                                kernel_size,
                                                stride,
                                                padding,
                                                padding_mode,
                                                dilation,
                                                **kwargs)
        print("Encoder", self.encoder)
        conv_inputs = u.process_convolutional_inputs([kernel_size, stride, padding, dilation])
        kernel_size, stride, padding, dilation = conv_inputs
        reversed_conv_params = u.reverse_lists([kernel_size,
                                                stride,
                                                padding,
                                                dilation])
        self.decoder_bridge = nn.Linear(latent_dim, num_filters[-1])
        self.decoder_bridge_1 = nn.Linear(num_filters[-1], np.prod(self.encoder.out_shape))
        self.last_conv = nn.Conv2d(num_filters[0],
                                   input_shape[0],
                                   kernel_size=1,
                                   dilation=1,
                                   padding=0,
                                   padding_mode="zeros",
                                   stride=1)
        out_shape = self.encoder.all_shapes
        new_out = list(reversed(copy.deepcopy(out_shape)))
        out_shape_1 = []
        for i in range(len(new_out)):
            out_shape_1.append([new_out[i][1], new_out[i][2]])
        self.decoder = e.ConvDecoder(reverse_layer_sizes,
                                     input_shape=self.encoder.out_shape,
                                     kernel_size=reversed_conv_params[0],
                                     output_layers=out_shape_1[1:],
                                     stride=reversed_conv_params[1],
                                     padding=reversed_conv_params[2],
                                     padding_mode=padding_mode,
                                     dilation=reversed_conv_params[3],
                                     **kwargs)
