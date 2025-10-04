#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 17:12:45 2025

@author: annabel
"""
import pickle
import numpy as np

with open('FINAL_PRED_BEST.pkl','rb') as f:
    params = pickle.load(f)['params']['params']

param_count = 0

for key, subdict in params.items():
    for key, mat in subdict.items():
        param_count += mat.size

print(param_count)
