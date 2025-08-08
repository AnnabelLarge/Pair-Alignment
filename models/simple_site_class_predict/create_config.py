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


def create_config(load_all: bool,
                  times_from: str):
    ### 
    if load_all:
        filenames_dict = OrderedDict({"times_file": "[STR, None]",
                                      "exch": "[STR, None]",
                                      "class_probs": "[STR, None]",
                                      "rate_mult": "[STR, None]",
                                      "equl_dist": "[STR, None]",
                                      "(if tkf indel model) tkf_params_file": "[STR, None]",
                                      "(if no indel model) geom_length_params_file": "[STR, None]"})
        to_add = OrderedDict({"filenames": filenames_dict})
    
    elif not load_all:
        to_add = OrderedDict({"exchange_range": "[min=FLOAT, max=FLOAT]",
                              "(if rate_mult_activation==bound_sigmoid) rate_mult_range": "[min=FLOAT, max=FLOAT]",
                              "lambda_range": "[min=FLOAT, max=FLOAT]",
                              "offset_range": "[min=FLOAT, max=FLOAT]",
                              "(if TKF92) r_range": "[min=FLOAT, max=FLOAT]",
                              
                              "init_lambda_offset_logits": "[FLOAT, FLOAT]",
                              "init_r_extend_logits": "[FLOAT]*num_mixtures",
                              
                              "filenames": {"exch": "[STR]",
                                            "times_file": "[STR, None]"} })
    
    out = OrderedDict( {"load_all": "[BOOL]",
                        "num_domain_mixtures": "[INT]",
                        "num_fragment_mixtures": "[INT]",
                        "num_site_mixtures": "[INT]",
                        "k_rate_mults": "[INT]",
                        
                         "LINEBREAK401": "",
                         "subst_model_type": '[STR="hky85","gtr"]',
                         "norm_rate_matrix": '[BOOL]',
                         "norm_rate_mults": '[BOOL]',
                         
                         "LINEBREAK402": "",
                         'indel_model_type': '[ None, STR=["tkf91", "tkf92"] ]',
                         "tkf_function": '[STR="regular_tkf","approx_tkf","switch_tkf"]',
                         
                         "LINEBREAK403": "",
                         "times_from": "[ STR: ('geometric', 't_array_from_file', 't_per_sample') ]",
                         "(if marginalizing over times) exponential_dist_param": "[FLOAT]",
                         "(if marginalizing over times) min_time": "[FLOAT]",
                         "(if times_from == 'geometric') t_grid_center": "[FLOAT]",
                         "(if times_from == 'geometric') t_grid_step": "[FLOAT]",
                         "(if times_from == 'geometric') t_grid_num_steps": "[INT]",
                         
                         "LINEBREAK404": ""
                         
                         })
    out = concat_dicts( [out, to_add] )
    return out
                  
