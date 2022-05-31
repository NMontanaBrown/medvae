# coding=utf-8

"""
"""

from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import medvae.models.utils as u

## Classic FC Encoder-Decoders
class Encoder(nn.Module):
    def __init__(self,
                 num_filters:List[int],
                 input_shape:List[int],
                 **kwargs):
        """
        Instantiate the Encoder module.
        :param layer_sizes: List[int], number of
                            fully connected neurons
                            in each fully connected layer.
        :param input_shape: List[int], Ch first input shape to
                            Encoder.
        """
        super(Encoder, self).__init__(**kwargs)
        self.layers = nn.ModuleList([nn.Linear(item,
                                               num_filters[i+1])
                                     for i, item in enumerate(num_filters)
                                     if i<(len(num_filters)-1)])
        self.input_shape = input_shape

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for decoder. Pass inputs
        through layers, activate using sigmoid,
        reshape.
        :param x: torch.Tensor, [B, CH, ...] input.
        :return:
        """
        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

class Decoder(Encoder):
    """
    Decoder module: same construction process
    as the Encoder, with a modified forward pass.
    """

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for decoder. Pass inputs
        through layers, activate using sigmoid,
        reshape.
        :param x: torch.Tensor, [B, N] input.
        :return: torch.Tensor, [B, *input_shape].
        """
        for i, layer in enumerate(self.layers):
            if i < (len(self.layers)-1):
                x = F.relu(layer(x))
            else:
                x = torch.sigmoid(layer(x))
        return x.reshape((-1, *self.input_shape))

## Convolutional Encoder-Decoders
class ConvEncoder(nn.Module):
    def __init__(self,
                 num_filters:List[int],
                 input_shape:List[int],
                 kernel_size:Union[List[int], List[list]],
                 stride:Union[List[int], List[list]],
                 padding:Union[List[int], List[list]],
                 padding_mode:List[str],
                 dilation:Union[List[int], List[list]]=[1,1],
                 transpose:bool=False,
                 **kwargs):
        """
        Configurable convolutional encoder.
        :param num_filters: List[int], num filters for
                            each convolutional layer.
        :param input_shape: List[int], input shape to
                            first conv layer.
        :param kernel_size: List[int], kernel_size
        :param stride: List[int]
        :param padding: List[int]
        :param padding_mode: str
        :raises:
        """
        super(ConvEncoder, self).__init__(**kwargs)
        # Check all inputs have same dims
        input_lists = [num_filters, kernel_size, stride, padding, dilation]
        u.check_input_lists(input_lists)
        # Check that the input are in the right format.
        conv_inputs = u.process_convolutional_inputs([kernel_size, stride, padding, dilation])
        kernel_size, stride, padding, dilation = conv_inputs

        # Calc output_shape
        out_shape, all_shapes = u.calculate_size_series_conv(num_filters,
                                                    input_shape,
                                                    kernel_size,
                                                    stride,
                                                    padding,
                                                    dilation,
                                                    return_all_shapes=True)
        self.all_shapes = all_shapes
        self.out_shape = out_shape
        # Construct the convolutional layers
        self.layers = nn.ModuleList()
        for i, item in enumerate(num_filters):
            if i == 0:
                self.layers.append(nn.Conv2d(input_shape[0],
                                                item,
                                                kernel_size=kernel_size[i],
                                                stride=stride[i],
                                                padding=padding[i],
                                                padding_mode=padding_mode[i],
                                                dilation=dilation[i]))
            else:
                self.layers.append(nn.Conv2d(num_filters[i-1],
                                                item,
                                                kernel_size=kernel_size[i],
                                                stride=stride[i],
                                                padding=padding[i],
                                                padding_mode=padding_mode[i],
                                                dilation=dilation[i]))


    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for convolutional encoder.
        :param x: torch.Tensor, [B, Ch, ...]
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

class ConvDecoder(nn.Module):
    def __init__(self,
                 num_filters:List[int],
                 input_shape:List[int],
                 kernel_size:Union[List[int], List[list]],
                 stride:Union[List[int], List[list]],
                 padding:Union[List[int], List[list]],
                 padding_mode:List[str],
                 dilation:Union[List[int], List[list]],
                 output_layers:List[list],
                 **kwargs):
        """
        """
        super(ConvDecoder, self).__init__(**kwargs)
        # Check all inputs have same dims
        input_lists = [num_filters, kernel_size, stride, padding, dilation]
        u.check_input_lists(input_lists)
        # Check that the input are in the right format.
        conv_inputs = u.process_convolutional_inputs([kernel_size, stride, padding, dilation])
        kernel_size, stride, padding, dilation = conv_inputs
        self.output_layers = output_layers
        self.layers = nn.ModuleList()
        for i, item in enumerate(num_filters):
            if i == 0:
                self.layers.append(nn.ConvTranspose2d(input_shape[0],
                                                       item,
                                                       kernel_size=kernel_size[i],
                                                       stride=stride[i],
                                                       padding=padding[i],
                                                       padding_mode=padding_mode[i],
                                                       dilation=dilation[i]))
            else:
                self.layers.append(nn.ConvTranspose2d(num_filters[i-1],
                                                      item,
                                                      kernel_size=kernel_size[i],
                                                      stride=stride[i],
                                                      padding=padding[i],
                                                      padding_mode=padding_mode[i],
                                                      dilation=dilation[i]))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for convolutional encoder.
        Uses expected output layers to calculate
        correct output size.
        :param x: torch.Tensor, [B, Ch, ...]
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x, output_size=self.output_layers[i]))
        return x


## Variational Encoders
class VariationalEncoderBase(nn.Module):
    """
    """
    def __init__(self,
                 num_filters:List[int],
                 input_shape:List[int],
                 latent_dim:int,
                 **kwargs):
        """
        :param num_filters: List[int]
        :param input_shape: List[int]
        :param latent_dim: int
        """
        super(VariationalEncoderBase, self).__init__(num_filters,
                                                     input_shape,
                                                     **kwargs)
        self.num_filters = num_filters
        self.latent_dim_mu = nn.Linear(num_filters[-1], latent_dim)
        self.latent_dim_sigma = nn.Linear(num_filters[-1], latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self,
                            x_hat:torch.Tensor,
                            logscale:torch.Tensor,
                            x:torch.Tensor):
        """
        :param x_hat: torch.Tensor, reconstructed
                      batch from sampled latent dim.
        :param logscale: torch.Tensor, 0.0
        :param x: torch.Tensor, input data batch.
        """
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(x_hat, scale)

        # measure prob of seeing data x given recon x_hat
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(list(range(1, len(self.input_shape)+1))))
    
    def kl_divergence(self,
                      z:torch.Tensor,
                      mu:torch.Tensor,
                      std:torch.Tensor)->torch.Tensor:
        """
        KL-divergence function
        :param z: torch.Tensor, resampled latent vector.
        :param mu: torch.Tensor, mean 
        :param std: torch.Tensor, standard deviation
        :return: torch.Tensor, [B,]
        """
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z) # Q(z|x)
        log_pz = p.log_prob(z) # Proba z near normal dist

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

class FCVariationalEncoder(VariationalEncoderBase, Encoder):
    def __init__(self,
                 num_filters:List[int],
                 input_shape:List[int],
                 latent_dim:int,
                 **kwargs):
        """
        :param num_filters: List[int]
        :param input_shape: List[int]
        :param latent_dim: List[int]
        """
        super(FCVariationalEncoder, self).__init__(num_filters,
                                                   input_shape,
                                                   latent_dim,
                                                   **kwargs)
        self.out_shape = [latent_dim]

class ConvVariationalEncoder(VariationalEncoderBase, ConvEncoder):
    def __init__(self,
                 num_filters:List[int],
                 input_shape:List[int],
                 latent_dim:int,
                 kernel_size:Union[List[int], List[list]],
                 stride:Union[List[int], List[list]],
                 padding:Union[List[int], List[list]],
                 padding_mode:List[str],
                 dilation:Union[List[int], List[list]]=[1,1],
                 transpose:bool=False,
                 **kwargs):
        """
        :param num_filters: List[int]
        :param input_shape: List[int]
        :param latent_dim: int
        :param kernel_size: List[int]
        :param stride: List[int]
        :param padding: List[int]
        :param padding_mode: str
        """
        super(ConvVariationalEncoder, self).__init__(num_filters,
                                                   input_shape,
                                                   latent_dim,
                                                   **{"kernel_size":kernel_size,
                                                    "stride":stride,
                                                    "padding":padding,
                                                    "padding_mode":padding_mode,
                                                    "dilation":dilation,
                                                    "transpose":transpose,
                                                    **kwargs})
        self.linear_bridge = nn.Linear(np.prod(self.out_shape), num_filters[-1])
        self.input_shape = input_shape

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for convolutional encoder.
        :param x: torch.Tensor, [B, Ch, ...]
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        x = x.view(-1, np.prod(self.out_shape))
        x = F.relu(self.linear_bridge(x))
        return x
