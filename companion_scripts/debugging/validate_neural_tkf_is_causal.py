#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:17:00 2025

@author: annabel_large
"""
from cli.test_neural_tkf_model_is_causal import test_neural_tkf_model_is_causal
import os

mydir = 'configs'

for file in os.listdir(mydir):
    if file.startswith('CONFIG') and file.endswith('.json'):
        path = f'{mydir}/{file}'
        test_neural_tkf_model_is_causal(path)
