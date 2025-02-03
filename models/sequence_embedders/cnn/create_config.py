#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:13:23 2024

@author: annabel
"""
from collections import OrderedDict


def create_config(*args, **kwargs):
    print('only one module for CNN: ConvnetBlock')
    
    return OrderedDict( {"initial_embed_module": "[STR]",
                         
                         "LINEBREAK1":"",
                         
                         "kern_size_lst": "[list of INT]",
                         "hidden_dim": "[INT]",
                         "dropout": "[FLOAT=0.0]",
                         } )

