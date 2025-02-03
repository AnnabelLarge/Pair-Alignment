#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:47:05 2024

@author: annabel
"""
from collections import OrderedDict


def create_config(*args, **kwargs):
    return OrderedDict( {"initial_embed_module": "[STR]",
                         "first_block_module": "[STR]",
                         "subsequent_block_module": "[STR]",
                         
                         "LINEBREAK1":"",
                         
                         "num_blocks": "[INT]"
                         "num_heads": "[INT]",
                         "hidden_dim": "[INT]",
                         "dropout": "[FLOAT=0.0]",
                         } )

