# coding=utf-8

"""
Definition of AE using PyTorch Lightning
"""

import copy
from typing import List
import torch.nn as nn
import medvae.models.encoder as e
import pytorch_lightning as pl

class AE(pl.LightningModule):
    """
    Implements an Auto-Encoder (AE)
    architecture. 
    """
    def __init__(self,
                 layer_sizes:List[int],
                 input_size:List[int]):
        """
        :param layer_sizes:
        :param input_size:
        """
        super(AE, self).__init__()

        self.encoder = e.Encoder(layer_sizes, input_size)
        reverse_layer_sizes = copy.deepcopy(layer_sizes)
        reverse_layer_sizes.reverse()
        self.decoder = e.Decoder(reverse_layer_sizes, input_size)

    def forward(self, x):
        """
        """
        x = self.encoder(x)
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        """
        Custom training step for VAE
        """
        raise NotImplementedError()

    def configure_optimizers(self):
        """
        """
        raise NotImplementedError()
