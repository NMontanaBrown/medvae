# coding=utf-8

"""
Testing model construction and calling
"""

import pytest
import torch
from medvae.models import encoder as e
from medvae.models.ae import AE as V
from medvae.models.vae import ConvVAE


@pytest.mark.parametrize("layers_shapes,input_shape", [([10,5,4], [1, 5, 2],),
                                                       ([100,50,30,20,3], [1, 10, 10])
                                                      ])
def test_AE_build(layers_shapes, input_shape):
    test_V = V(layers_shapes, input_shape)
    output = test_V.forward(torch.ones((1,*input_shape)))
    assert output.shape == (1, *input_shape)

def test_convVae():
    """
    """
    enc = ConvVAE(num_filters=[6,5,3],
                  input_shape=[1, 50, 50],
                  latent_dim=2,
                  kernel_size=[[1,1],[2,2], [3,3]],
                  stride=[2,2,2],
                  padding=[0,0,0],
                  padding_mode=["zeros", "zeros", "zeros"],
                  dilation=[1,1,1])
    out = enc.training_step([torch.ones((1,1,50,50)), None], 0)
    assert list(out.shape) == [1]
    enc = ConvVAE(num_filters=[7,6,5,3],
                  input_shape=[1, 100, 40],
                  latent_dim=2,
                  kernel_size=[[2,1], [1,1],[2,2], [3,3]],
                  stride=[1, 2,2,2],
                  padding=[0, 0,0,0],
                  padding_mode=["zeros", "zeros", "zeros", "zeros"],
                  dilation=[2,1,1,1])
    out = enc.training_step([torch.ones((1,1,100,40)), None], 0)
    assert list(out.shape) == [1]
