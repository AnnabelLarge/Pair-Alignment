#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:33:12 2024

@author: annabel
"""
from collections import OrderedDict


def concat_dicts(dict_lst):
    out = OrderedDict()
    for d in dict_lst:
        out = OrderedDict({**out, **d})
    return out


def create_config(preset_name):
    assert preset_name in ['base_hmm_load_all',
                           'base_hmm_fit_indel_params',
                           'local_exch_equilibr',
                           'local_exch_equilibr_r',
                           'all_local']
    
    ### all global, load from file
    if preset_name == 'base_hmm_load_all':
        exchang_config = OrderedDict( {'load_from_file': '[STR]',
                                       'unit_norm_rate_matrix': 'true'} )
        equilibr_config = OrderedDict( {} )
        indels_config = OrderedDict( {'load_from_file': '[STR]',
                                      'tkf_err': '[FLOAT=1e-4]'} )
    
    ### all global, only fit indel params
    elif preset_name == 'base_hmm_fit_indel_params':
        exchang_config = OrderedDict( {'load_from_file': '[STR]',
                                       'unit_norm_rate_matrix': 'true'} )
        equilibr_config = OrderedDict( {} )
        indels_config = OrderedDict( {'tkf_err': '[FLOAT=1e-4]'} )
    
    ### local exchange and equilibriums
    elif preset_name in ['local_exch_equilibr',
                         'local_exch_equilibr_r',
                         'all_local']:
        exchang_config = OrderedDict( {'use_anc_emb': '[BOOL]',
                                       'use_desc_emb': '[BOOL]',
                                       'layer_sizes': '[LIST[INTS]]',
                                       'dropout': '[FLOAT=0.0]',
                                       'exchange_range': '[min: FLOAT=1e-4, max: FLOAT=10]',
                                       'avg_pool': 'false',
                                       } )
        
        equilibr_config = OrderedDict( {'use_anc_emb': '[BOOL]',
                                       'use_desc_emb': '[BOOL]',
                                       'layer_sizes': '[LIST[INTS]]',
                                       'dropout': '[FLOAT=0.0]',
                                       'avg_pool': 'false',
                                       } )
        
        ### global indel params
        if preset_name == 'local_exch_equilibr':
            indels_config = OrderedDict( {'manual_init': '[BOOL]',
                                          '(if manual_init) load_from_file': '[STR]',
                                          'tkf_err': '[FLOAT=1e-4]',
                                          'lamdba_range': '[min: FLOAT=tkf_err, max: FLOAT=3]',
                                          'offset_range': '[min: FLOAT=tkf_err, max: FLOAT=0.333]',
                                          '(if TKF92) r_range': '[min: FLOAT=tkf_err, max: FLOAT=0.8]'} )
        
        ### local indel params
        elif preset_name == 'all_local':
            indels_config = OrderedDict( {'use_anc_emb': '[BOOL]',
                                           'use_desc_emb': '[BOOL]',
                                           'layer_sizes': '[LIST[INTS]]',
                                           'dropout': '[FLOAT=0.0]',
                                           'avg_pool': 'false',
                                           'manual_init': 'false',
                                           'tkf_err': '[FLOAT=1e-4]',
                                           'lamdba_range': '[min: FLOAT=tkf_err, max: FLOAT=3]',
                                           'offset_range': '[min: FLOAT=tkf_err, max: FLOAT=0.333]',
                                           '(if TKF92) r_range': '[min: FLOAT=tkf_err, max: FLOAT=0.8]'} )
    
    
    return OrderedDict( {"times_from": "[ STR: ('geometric', 't_array_from_file', 'one_time_per_sample') ]",
                         "exponential_dist_param": "[FLOAT]",
                         'indel_model_type': '[STR="tkf91", "tkf92"]',
                         'use_precomputed_indices: [BOOL]',
                         
                         "LINEBREAK401": "",
                         
                         "t_grid_center": "[FLOAT; use if times_from = geometric]",
                         "t_grid_step": "[FLOAT; use if times_from = geometric]",
                         "t_grid_num_steps": "[INT; use if times_from = geometric]",
                         
                         "LINEBREAK402": "",
                         
                         "times_file": "[STR]; use if times_from = t_array_from_file",
                         
                         "LINEBREAK4": "",
                         
                         "exchang_config": exchang_config,
                         
                         "LINEBREAK5": "",
                         
                         "equilibr_config": equilibr_config,
                         
                         "LINEBREAK6": "",
                         
                         "indels_config": indels_config
                         } )
