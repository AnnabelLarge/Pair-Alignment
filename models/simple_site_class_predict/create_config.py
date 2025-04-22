#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:42:44 2025

@author: annabel
"""
from collections import OrderedDict


def concat_dicts(dict_lst):
    out = OrderedDict()
    for d in dict_lst:
        out = OrderedDict({**out, **d})
    return out


def create_config(load_all: bool):
    if load_all:
        filenames_dict = OrderedDict({"times_file": "[STR, None]",
                                      "exch": "[STR, None]",
                                      "class_probs": "[STR, None]",
                                      "rate_mult": "[STR, None]",
                                      "equl_dist": "[STR, None]",
                                      "tkf_params_file": "[STR, None]"})
        to_add = OrderedDict({"filenames": filenames_dict})
    
    elif not load_all:
        to_add = OrderedDict({"exchange_range": "[min=FLOAT, max=FLOAT]",
                              "(if rate_mult_activation==bound_sigmoid) rate_mult_range": "[min=FLOAT, max=FLOAT]",
                              "lambda_range": "[min=FLOAT, max=FLOAT]",
                              "(if TKF92) offset_range": "[min=FLOAT, max=FLOAT]",
                              })
    
    else:
        to_add = {}
    
    out = OrderedDict( {"load_all_params": "[BOOL]",
                        "(if indp sites) preset_name": "[ STR: ('load', 'fit_rate_mult_only', 'fit_rate_mult_and_matrix') ]",
                         "times_from": "[ STR: ('geometric', 't_array_from_file') ]",
                         "exponential_dist_param": "[FLOAT]",
                         'indel_model_type': '[STR="tkf91", "tkf92"]',
                         "num_emit_site_classes": "[INT]",
                         "num_tkf_site_classes": "[INT, 1 if indp site classes]",
                         "tkf_err": "[FLOAT = 1e4]",
                         "rate_mult_activation": '[STR="bound_sigmoid", "softplus"]',
                         
                         "LINEBREAK401": "",
                         
                         "(if times_from == 'geometric') t_grid_center": "[FLOAT]",
                         "(if times_from == 'geometric') t_grid_step": "[FLOAT]",
                         "(if times_from == 'geometric') t_grid_num_steps": "[INT]"
                         
                         })
    out = concat_dicts( [out, to_add] )
    return out
                  
