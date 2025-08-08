#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 18:42:32 2025

@author: annabel_large
"""
import pickle
import numpy as np

with open(f'PARAMS-DICT_test-set_pt0_tkf91_indel_params.pkl','rb') as f:
    old_dict = pickle.load(f)

new_dict = old_dict.copy()
new_dict['r_extend'] = np.zeros( (1,1) )

with open(f'PARAMS-DICT_tkf92-equl_params.pkl','wb') as g:
    pickle.dump(new_dict, g)
