#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:57:26 2024

@author: annabel
"""
from collections import OrderedDict


def create_config(*args, **kwargs):
    return OrderedDict({"initial_embed_module": "[STR]",
                       "first_block_module": "[STR]",
                       "subsequent_block_module": "[STR]",
                         
                       "LINEBREAK1":"",
                       
                       "n_layers": "[LIST of int]",
                       "return_final_carry": "[BOOL=false]",
                       "hidden_dim": "[INT]",
                       "dropout": "[FLOAT=0.0]"})
    
