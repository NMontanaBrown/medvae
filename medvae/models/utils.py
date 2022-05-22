# coding=utf-8

"""
Utility functions for model construction.
"""

import copy
from math import floor
from typing import List

def check_input_lists(input_list:List[list]):
    """
    Checks if list of lists have same length.
    :param input_list: List[list]
    :raise: ValueErrr if all sublists not same length
    """
    iter_lists = iter(input_list)
    the_len = len(next(iter_lists))
    if not all(len(l) == the_len for l in iter_lists):
        raise ValueError('Parameter lists not the same length.')

def reverse_lists(input_list:List[list]):
    """
    Utility function that reverses
    the elements in series of lists.
    :param input_list: List[list]
    :raise: ValueError if all sublists not same length
    """
    check_input_lists(input_list)
    new_list = []
    for item in input_list:
        reverse_list = copy.deepcopy(item)
        new_list.append(reverse_list.reverse())
    return new_list

def calculate_final_layer_size(num_filters:int,
                               input_shape:List[int],
                               kernel_size:List[int],
                               stride:List[int],
                               padding:List[int],
                               dilation:List[int]=[1,1]):
    """
    Function to calculate the final
    tensor size through a convolutional
    layer given an input shape and conv
    parameters.
    :param num_filters: int
    :param input_shape: List[int], [Ch, H, W]
    :param kernel_size: List[int]
    :param stride: List[int], len(2,)
    :param padding: List[int], len(2,)
    :return:
    """
    H_out = floor((input_shape[1] + (2*padding[0]) - dilation[0]*(kernel_size[0]-1) -1) / (stride[0])) + 1
    W_out = floor((input_shape[2] + (2*padding[1]) - dilation[1]*(kernel_size[1]-1) -1) / (stride[1])) + 1
    return [num_filters, H_out, W_out]

def calculate_size_series_conv(num_filters:List[int],
                               input_shape:List[list],
                               kernel_size:List[list],
                               stride:List[list],
                               padding:List[list],
                               dilation:List[list]):
    """
    Function to calculate the final
    tensor size through a convolutional
    encoder given an input shape and conv
    parameters.
    """
    check_input_lists([num_filters,
                      kernel_size,
                      stride,
                      padding,
                      dilation])
    out_shape = input_shape
    for i, _ in enumerate(kernel_size):
        out_shape = calculate_final_layer_size(num_filters[i],
                                               out_shape,
                                               kernel_size[i],
                                               stride[i],
                                               padding[i],
                                               dilation[i])
    return out_shape