#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:49:08 2024

@author: annabel
"""
from collections import OrderedDict


def create_config(*args, **kwargs):
    return OrderedDict({"add_prev_alignment_info": "[BOOL]",
                        "layer_sizes": "[list of INT]",
                        "normalize_inputs": "[BOOL]",
                        "dropout": "[FLOAT=0.0]"})

