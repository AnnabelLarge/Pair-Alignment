#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:05:55 2024

@author: annabel
"""
from collections import OrderedDict


def create_config(bidirect: bool,
                  *args, 
                  **kwargs):
    out = OrderedDict( {"initial_embed_module": "[STR]",
                         "first_block_module": "[STR]",
                         "subsequent_block_module": "[STR]",
                         
                         "LINEBREAK1":"",
                         
                         "num_blocks": "[INT]",
                         "expansion_factor": "[INT]",
                         "hidden_dim": "[INT]",
                         "dropout": "[FLOAT=0.0]",
                         
                         "LINEBREAK3":"",
                        
                         "ssm_hidden_features": "[INT]=16",
                         "dt_rank": "[ (INT, 'auto')='auto' ]",
                         "dt_proj": "[BOOL]=true",
                         "ssm_shift_conv_size": "[INT]=3",
                         "dt_min": "[FLOAT]=0.001",
                         "dt_max": "[FLOAT]=0.1"
                         } )
    
    if bidirect:
        to_add = OrderedDict({"LINEBREAK4":"",
                              "tie_in_proj": "[BOOL]=false",
                              "tie_gate": "[BOOL]=false"})
        out = OrderedDict({**out, **to_add})
    
    return out

    