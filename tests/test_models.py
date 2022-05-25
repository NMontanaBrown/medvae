# coding=utf-8

"""
Testing model construction and calling
"""

import pytest
import torch
from medvae.models.utils import (
    calculate_size_series_conv,
    check_input_lists,
    calculate_final_layer_size,
    process_list_params,
    reverse_lists)
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

@pytest.mark.parametrize("test_inputs, expected_outputs", 
                                        [([[1,2,3], [1, 2, 3], [1, 2, 3],],
                                          [[3,2,1], [3,2,1], [3,2,1],]),
                                         ([[1, 2], [1, 2], [1, 2]],
                                          [[2, 1], [2, 1], [2, 1]]),
                                         ])
def test_check_lists_reversed(test_inputs, expected_outputs):
    """
    Check correctly reversed.
    """
    out = reverse_lists(test_inputs)
    assert out == expected_outputs

@pytest.mark.parametrize("test_inputs, expected_outputs", 
                                        [([1,2,3],
                                          [[1,1], [2,2], [3,3]]),
                                         ([1,[2,2],3],
                                          [[1,1], [2,2], [3,3]]),
                                         ])
def test_check_process_lists(test_inputs, expected_outputs):
    """
    Check lists correctly processed
    """
    out = process_list_params(test_inputs)
    assert out == expected_outputs


@pytest.mark.parametrize("test_inputs", 
                                        [([[1,1], [2,"a"], [3,3]]),
                                         ([1,[2,2],"a"]),
                                         ])
def test_check_process_lists_value_error(test_inputs):
    """
    Check value error raised if incorrect type in
    list.
    """
    with pytest.raises(ValueError):
        process_list_params(test_inputs)

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

@pytest.mark.parametrize("num_filters,input_shape,kernel_size,stride,padding,padding_mode,dilation",
                         [([3,6,9], [1, 30, 30],[2,2,2],[1,1,1],[0,0,0],["zeros" for i in range(3)], [1,1,1]),
                         ([10,20,30,40], [1, 100, 100],[5,5,2,2],[1,1,1,1],[0,0,0,0],["zeros" for i in range(4)], [1,1,1,1]),])
def test_ConvEncoder_build(num_filters,input_shape,kernel_size,stride,padding,padding_mode,dilation):
    test_CE = CE(num_filters,input_shape,kernel_size,stride,padding,padding_mode,dilation)
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

@pytest.mark.parametrize("num_filters,input_shape,kernel_size,stride,padding,dilation,return_all_shapes",
                         [([3, 6, 9],
                           [1, 30, 30],
                           [[2,2] for i in range(3)],
                           [[2,2] for i in range(3)],
                           [[0,0] for i in range(3)],
                           [[1,1] for i in range(3)],
                           False),
                           ([3, 6, 9, 12],
                           [1, 100, 100],
                           [[5,5] for i in range(4)],
                           [[2,3] for i in range(4)],
                           [[0,1] for i in range(4)],
                           [[1,1] for i in range(4)],
                           False),
                           ([3, 6, 9],
                           [1, 30, 30],
                           [[2,2] for i in range(3)],
                           [[2,2] for i in range(3)],
                           [[0,0] for i in range(3)],
                           [[1,1] for i in range(3)],
                           True),
                           ([3, 6, 9, 12],
                           [1, 100, 100],
                           [[5,5] for i in range(4)],
                           [[2,3] for i in range(4)],
                           [[0,1] for i in range(4)],
                           [[1,1] for i in range(4)],
                           True)])
def test_calc_layer_size_series(num_filters,input_shape,kernel_size,stride,padding,dilation,return_all_shapes):
    if return_all_shapes:
        out_shape, all_shapes = calculate_size_series_conv(num_filters, input_shape, kernel_size, stride,padding, dilation,return_all_shapes)
    else:
        out_shape = calculate_size_series_conv(num_filters, input_shape, kernel_size, stride,padding, dilation,return_all_shapes)
    all_shapes_t = []
    for i, _ in enumerate(num_filters):
        if i ==0:
            conv= torch.nn.Conv2d(input_shape[0],
                                  num_filters[i],
                                  kernel_size[i],
                                  stride[i],
                                  padding[i],
                                  dilation[i])
            out_tens = conv(torch.ones((1, *input_shape)))
            all_shapes_t.append(list(out_tens.shape[1:]))
        else:
            conv = torch.nn.Conv2d(num_filters[i-1],
                                    num_filters[i],
                                    kernel_size[i],
                                    stride[i],
                                    padding[i],
                                    dilation[i])
            out_tens = conv(out_tens)
            all_shapes_t.append(list(out_tens.shape[1:]))
    out_shape_true = out_tens.shape
    assert [1,] + out_shape == list(out_shape_true)
    if return_all_shapes:
        assert all_shapes[1:] == all_shapes_t
