#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:49:08 2024

@author: annabel
"""
from collections import OrderedDict


def create_config(*args, **kwargs):
    return OrderedDict({"layer_sizes": "[list of INT]",
                        "normalize_inputs": "[BOOL]",
                        "use_bias": "[BOOL]",
                        "dropout": "[FLOAT=0.0]"})

