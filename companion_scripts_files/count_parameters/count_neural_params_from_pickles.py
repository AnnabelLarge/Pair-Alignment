#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 17:41:43 2025

@author: annabel
"""
import pickle
import numpy as np

def flatten_dict(d, parent_key="", sep="."):
    """
    Recursively flattens a nested dictionary.

    Args:
        d (dict): Dictionary to flatten
        parent_key (str): Current prefix for keys
        sep (str): Separator between nested keys

    Returns:
        dict: Flattened dictionary
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def count_params(file):
    # load
    with open(file,'rb') as f:
        params = pickle.load(f)['params']['params']
    params = flatten_dict(params)
    
    c = 0
    for key, mat in params.items():
        c += mat.size
    return c

anc_params = count_params(f'ANC_ENC_BEST.pkl')
desc_params = count_params(f'DESC_DEC_BEST.pkl')
final_params = count_params(f'FINAL_PRED_BEST.pkl')

param_count = anc_params + desc_params + final_params
