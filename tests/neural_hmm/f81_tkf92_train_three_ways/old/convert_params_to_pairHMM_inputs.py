#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 18:21:22 2025

@author: annabel_large


"""
import pickle
import numpy as np

### read
in_file = 'RESULTS_nonlocal_params/out_arrs/test-set_pt0_ARRS.pkl'

def load_params(file):
    with open(file,'rb') as f:
        return pickle.load(f)

params = load_params(in_file)

indel_model_params = params['INDEL_MODEL_PARAMS']
lam = np.array( indel_model_params['lambda'] )[0,0]
mu = np.array( indel_model_params['mu'] )[0,0]
r_extend = np.array( indel_model_params['r_extend'] )[0,0]
del indel_model_params

subs_model_params = params['SUBS_MODEL_PARAMS']
rate_multiplier = np.array( subs_model_params['rate_multiplier'] )
del subs_model_params

equl = np.array( np.exp( params['LOGPROB_EMIT_INDEL'] ) )[0,0,:]
del params


### output
tkf_params_dict = {'lambda': lam,
                   'mu': mu,
                   'r_extend': r_extend[None]}
with open('tkf_params_dict.pkl','wb') as g:
    pickle.dump(tkf_params_dict, g)

with open(f'equl.npy','wb') as g:
    np.save(g, equl)
    

