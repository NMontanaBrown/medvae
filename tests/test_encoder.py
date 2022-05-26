# coding=utf-8

"""
Testing encoder decoder construction
"""

import pytest
import torch
import copy
from medvae.models import encoder as e
from medvae.models.encoder import ConvEncoder as CE
from medvae.models.encoder import ConvDecoder, ConvVariationalEncoder

@pytest.mark.parametrize("layers_shapes,input_shape,out_shape", [([10, 5, 4], [1, 5, 2], (1,4)),
                                                                 ([100, 50, 20, 3], [1, 10, 10], (1, 3))
                                                                 ])
def test_encoder_build(layers_shapes, input_shape, out_shape):
    test_e = e.Encoder(layers_shapes, input_shape)
    out = test_e.forward(torch.ones((1, *input_shape)))
    assert out.shape == out_shape

@pytest.mark.parametrize("layers_shapes,input_shape,dummy_shape,out_shape", [([4, 5, 10], [1, 5, 2], [4], (1,1,5,2)),
                                                                 ([3, 20, 50, 100], [1, 10, 10], [3], (1, 1, 10, 10))
                                                                 ])
def test_decoder_build(layers_shapes, input_shape, dummy_shape, out_shape):
    test_e = e.Decoder(layers_shapes, input_shape)
    out = test_e.forward(torch.ones((1, *dummy_shape)))
    assert out.shape == out_shape


@pytest.mark.parametrize("num_filters,input_shape,kernel_size,stride,padding,padding_mode,dilation",
                         [([3,6,9], [1, 30, 30],[2,2,2],[1,1,1],[0,0,0],["zeros" for i in range(3)], [1,1,1]),
                         ([10,20,30,40], [1, 100, 100],[5,5,2,2],[1,1,1,1],[0,0,0,0],["zeros" for i in range(4)], [1,1,1,1]),])
def test_ConvEncoder_build(num_filters,input_shape,kernel_size,stride,padding,padding_mode,dilation):
    test_CE = CE(num_filters,input_shape,kernel_size,stride,padding,padding_mode,dilation)
    output = test_CE.forward(torch.ones(1, *input_shape))


def test_ConvDecoder_build():
    enc = CE(num_filters=[6,5,3],
                                 input_shape=[1, 50, 50],
                                 kernel_size=[[2,2],[2,2], [2,2]],
                                 stride=[2,2,2],
                                 padding=[0,0,0],
                                 padding_mode=["zeros", "zeros", "zeros"],
                                 dilation=[1,1,1])
    out_shape = enc.all_shapes
    out = enc(torch.ones(4, 1, 50, 50))
    new_out = list(reversed(copy.deepcopy(out_shape)))
    out_shape_1 = []
    for i in range(len(new_out)):
        out_shape_1.append([new_out[i][1], new_out[i][2]])
    dec = ConvDecoder(num_filters=[3,5,6],
                      input_shape=list(out.shape[1:]),
                      kernel_size=[[2,2],[2,2], [2,2]],
                      stride=[2,2,2],
                      padding=[0,0,0],
                      padding_mode=["zeros", "zeros", "zeros"],
                      dilation=[1,1,1],
                       output_layers=out_shape_1[1:])
    out_test = dec(torch.ones((4,3,6,6)))
    assert out_test.shape == (4,6,50,50)
    
def test_ConvVarEnc_build():
    """
    """
    enc = ConvVariationalEncoder(num_filters=[6,5,3],
                                 input_shape=[1, 50, 50],
                                 latent_dim=2,
                                 kernel_size=[[2,2],[2,2], [2,2]],
                                 stride=[2,2,2],
                                 padding=[0,0,0],
                                 padding_mode=["zeros", "zeros", "zeros"],
                                 dilation=[1,1,1])
    out = enc(torch.ones(4, 1, 50, 50))
    assert out.shape == (4, 3)