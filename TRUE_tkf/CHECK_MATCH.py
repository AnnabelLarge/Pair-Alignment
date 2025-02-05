#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:00:52 2025

@author: annabel
"""
import numpy as np


pred = 'RESULTS_tkf91_load_params/out_arrs/AFTER-UPDATE_logprob_transits.npy'
true = 'TRUE_tkf/TRUE_jtkf91_mat.npy'

with open(pred, 'rb') as f:
    pred_mat = np.load(f)

with open(true, 'rb') as f:
    true_mat = np.load(f)


# for i,param in enumerate(['alpha', 'beta', 'gamma']):
#     pred = f'RESULTS_tkf91_load_params/out_arrs/AFTER-UPDATE_log_{param}.npy'
    
#     with open(pred, 'rb') as f:
#         pred_mat = np.load(f)
    
#     print(f'Predicted {param}: {pred_mat}')
    
#     true = 'TRUE_tkf/TRUE_tkf_alpha_beta_gamma.npy'
    
#     with open(true,'rb') as f:
#         true_mat = np.log( np.load(f)[i] )
    
#     print(f'True {param}: {true_mat}')
#     print()
    
