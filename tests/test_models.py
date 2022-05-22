# coding=utf-8

"""
Testing model construction and calling
"""

import pytest
import torch
from medvae.models.utils import calculate_size_series_conv, check_input_lists, calculate_final_layer_size
from medvae.models import encoder as e
from medvae.models.ae import AE as V
from medvae.models.encoder import ConvEncoder as CE

@pytest.mark.parametrize("test_inputs", [[[1,2], [1, 2, 3], [1, 2, 3]],
                                         [[1, 2, 3], [1, 2, 3], [1, 2]],
                                         [[1, 2, 3], [1, 2], [1, 2, 3]]
                                         ])
def test_check_lists_value_error(test_inputs):
    """
    Check fails correctly with lists different length
    """
    with pytest.raises(ValueError):
        check_input_lists(test_inputs)

@pytest.mark.parametrize("test_inputs", [[[1,2,3], [1, 2, 3], [1, 2, 3]],
                                         [[1, 2], [1, 2], [1, 2]],
                                         [[1], [1], [1], [1]]
                                         ])
def test_check_lists(test_inputs):
    """
    Check passes if lists same length.
    """

    check_input_lists(test_inputs)

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

@pytest.mark.parametrize("layers_shapes,input_shape", [([10,5,4], [1, 5, 2],),
                                                       ([100,50,30,20,3], [1, 10, 10])
                                                      ])
def test_AE_build(layers_shapes, input_shape):
    test_V = V(layers_shapes, input_shape)
    output = test_V.forward(torch.ones((1,*input_shape)))
    assert output.shape == (1, *input_shape)

@pytest.mark.parametrize("num_filters,input_shape,kernel_size,stride,padding,padding_mode",
                         [([3,6,9], [1, 30, 30],[2,2,2],[1,1,1],[0,0,0],"zeros"),
                         ([10,20,30,40], [1, 100, 100],[5,5,2,2],[1,1,1,1],[0,0,0,0],"zeros"),])
def test_ConvEncoder_build(num_filters,input_shape,kernel_size,stride,padding,padding_mode):
    test_CE = CE(num_filters,input_shape,kernel_size,stride,padding,padding_mode)
    output = test_CE.forward(torch.ones(1, *input_shape))

@pytest.mark.parametrize("num_filters,input_shape,kernel_size,stride,padding",
                         [([3], [1, 30, 30],[2,2],[2,2],[0,0]),
                          ([10], [1, 100, 100],[5,5],[2,2],[0,0]),
                          ([12], [1, 20, 20],[3,3],[1,1],[0,0]),
                          ([3], [1, 5, 5],[1, 2],[3,2],[0,0]),])
def test_calc_layer_size(num_filters,input_shape,kernel_size,stride,padding,):
    out_shape = calculate_final_layer_size(num_filters[0], input_shape, kernel_size, stride,padding)
    conv_layer = torch.nn.Conv2d(1,  num_filters[0],kernel_size,stride,padding,)
    out_shape_true = conv_layer(torch.ones((1, *input_shape)))
    assert [1,] + out_shape == list(out_shape_true.shape)

@pytest.mark.parametrize("num_filters,input_shape,kernel_size,stride,padding, dilation",
                         [([3, 6, 9],
                           [1, 30, 30],
                           [[2,2] for i in range(3)],
                           [[2,2] for i in range(3)],
                           [[0,0] for i in range(3)],
                           [[1,1] for i in range(3)]),
                           ([3, 6, 9, 12],
                           [1, 100, 100],
                           [[5,5] for i in range(4)],
                           [[2,3] for i in range(4)],
                           [[0,1] for i in range(4)],
                           [[1,1] for i in range(4)])])
def test_calc_layer_size_series(num_filters,input_shape,kernel_size,stride,padding,dilation):
    out_shape = calculate_size_series_conv(num_filters, input_shape, kernel_size, stride,padding, dilation)
    for i, _ in enumerate(num_filters):
        if i ==0:
            conv= torch.nn.Conv2d(input_shape[0],
                                  num_filters[i],
                                  kernel_size[i],
                                  stride[i],
                                  padding[i],
                                  dilation[i])
            out_tens = conv(torch.ones((1, *input_shape)))
        else:
            conv = torch.nn.Conv2d(num_filters[i-1],
                                    num_filters[i],
                                    kernel_size[i],
                                    stride[i],
                                    padding[i],
                                    dilation[i])
            out_tens = conv(out_tens)
    out_shape_true = out_tens.shape
    assert [1,] + out_shape == list(out_shape_true)
