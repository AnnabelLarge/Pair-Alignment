#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 18:48:23 2025

@author: annabel
"""
import numpy as np

vals = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 
        'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

str_mat = np.zeros((20,20)).astype(str)
for i,from_aa in enumerate(vals):
    for j,to_aa in enumerate(vals):
        str_mat[i,j] = f'{from_aa}_{to_aa}'
        
upper_tri = str_mat[np.triu_indices(str_mat.shape[0], k=1)]

with open(f'exch_upper_tri_labels.txt','w') as g:
    [g.write(elem + '\n') for elem in upper_tri]



