# coding=utf-8

"""
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import medvae.models.utils as u

## Classic FC Encoder-Decoders
class Encoder(nn.Module):
    def __init__(self,
                 layer_sizes:List[int],
                 input_shape:List[int]):
        """
        Instantiate the Encoder module.
        :param layer_sizes: List[int], number of
                            fully connected neurons
                            in each fully connected layer.
        :param input_shape: List[int], Ch first input shape to
                            Encoder.
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(item,
                                               layer_sizes[i+1])
                                     for i, item in enumerate(layer_sizes)
                                     if i<(len(layer_sizes)-1)])
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
                 latent_dims:int,
                 kernel_size:List[int],
                 stride:List[int],
                 padding:List[int],
                 padding_mode:str='zeros'):
        """
        Configurable convolutional encoder.
        :param num_filters: List[int], num filters for
                            each convolutional layer.
        :param input_shape: List[int], input shape to
                            first conv layer.
        :param latent_dims: int,
        :param kernel_size: List[int], kernel_size
        :param stride: List[int]
        :param padding: List[int]
        :param padding_mode: str
        :raises:
        """
        super(ConvEncoder, self).__init__()
        # Check all inputs have same dims
        input_lists = [num_filters, kernel_size, stride, padding]
        u.check_input_lists(input_lists)
        
        # Construct the convolutional layers
        self.layers = nn.ModuleList()
        for i, item in enumerate(num_filters):
            if i == 0:
                self.layers.append(nn.Conv2d(input_shape[0],
                                             item,
                                             kernel_size=kernel_size[i],
                                             stride=stride[i],
                                             padding=padding[i],
                                             padding_mode=padding_mode))
            else:
                self.layers.append(nn.Conv2d(num_filters[i-1],
                                             item,
                                             kernel_size=kernel_size[i],
                                             stride=stride[i],
                                             padding=padding[i],
                                             padding_mode=padding_mode))


    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        Forward pass for convolutional encoder.
        :param x: torch.Tensor, [B, Ch, ...]
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class ConvDecoder(ConvEncoder):

    def forward(self, x):
        """
        """
        raise NotImplementedError