# coding=utf-8

"""
Utility functions for model construction.
"""

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
