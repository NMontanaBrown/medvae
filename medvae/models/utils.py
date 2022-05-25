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
        new_list.append(item[::-1])
    return new_list

def process_list_params(input_list:list):
    """
    For a given list of unknown types, convert to
    len(2,) ints in the case of encountering an int,
    or maintain List[int] entry.
    :raise: ValueError
    :return: List[List[int]]
    """
    return_list = []
    for item in input_list:
        if isinstance(item, int):
            return_list.append([item for i in range(2)])
        elif isinstance(item, list) and all(isinstance(i, int) for i in item):
                return_list.append(item)
        else:
            raise ValueError("One of the items in input is neither an int or a List[int]")
    return return_list

def process_convolutional_inputs(input_list:List[list]):
    """
    For inputs to convolutional network builder,
    check and process each such that they are
    all List[List[int]]
    :param input_list: List[list], of unknown types.
    :raises:
    :return: List[List]
    """
    conv_inputs = []
    for item in input_list:
        conv_inputs.append(process_list_params(item))
    return conv_inputs

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
                               dilation:List[list],
                               return_all_shapes:bool=False):
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
    all_shapes = []
    all_shapes.append(input_shape)

    for i, _ in enumerate(kernel_size):
        out_shape = calculate_final_layer_size(num_filters[i],
                                               out_shape,
                                               kernel_size[i],
                                               stride[i],
                                               padding[i],
                                               dilation[i])
        all_shapes.append(out_shape)
    if return_all_shapes:
        return out_shape, all_shapes
    return out_shape
