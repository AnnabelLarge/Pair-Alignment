#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:06:50 2025

@author: annabel
"""
import numpy as np
import pandas as pd
import pickle

neural_dir = 'RESULTS_train_neuralTKF_force-global'

def write_matrix_to_npy(out_folder,
                        mat,
                        key):
    with open(f'{out_folder}/PARAMS-MAT_{key}.npy', 'wb') as g:
        np.save( g, mat )

def maybe_write_matrix_to_ascii(out_folder,
                                mat,
                                key):
    mat = np.squeeze(mat)
    if len(mat.shape) <= 2:
        np.savetxt( f'{out_folder}/ASCII_{key}.tsv', 
                    np.array(mat), 
                    fmt = '%.8f',
                    delimiter= '\t' )
        
params_file = f'{neural_dir}/out_arrs/test-set_pt0_ARRS.pkl'
with open(params_file,'rb') as f:
    param_dict = pickle.load(f)

### transitions: TKF92 indel parameters
lam = np.squeeze( np.array( param_dict['INDEL_MODEL_PARAMS']['lambda'] ) )
mu = np.squeeze( np.array( param_dict['INDEL_MODEL_PARAMS']['mu'] ) )
r = np.squeeze( np.array( param_dict['INDEL_MODEL_PARAMS']['r_extend'] ) )
cond_prob_transit = np.exp( np.squeeze( np.array(param_dict['LOGPROB_TRANSITS']) ) )


### emissions: for one f81, it's just equilibrium distribution
# equilibrium distribution
equl = np.exp( np.squeeze( np.array( param_dict['LOGPROB_EMIT_INDEL'] ) ) )
cond_prob_emit_match = np.exp( np.squeeze( np.array( param_dict['LOGPROB_EMIT_MATCH'] ) ) )


### write to files that are compatible with pairHMM code
# numpy arrays
# write_matrix_to_npy( out_folder = f'{neural_dir}/out_arrs',
#                       mat = ,
#                       key = 'test-set_pt0' )

write_matrix_to_npy( out_folder = f'{neural_dir}/out_arrs',
                      mat = cond_prob_transit,
                      key = 'test-set_pt0_conditional_prob_transit_matrix' )

write_matrix_to_npy( out_folder = f'{neural_dir}/out_arrs',
                      mat = cond_prob_emit_match,
                      key = 'test-set_pt0_cond_prob_emit_at_match' )

write_matrix_to_npy( out_folder = f'{neural_dir}/out_arrs',
                      mat = equl,
                      key = 'test-set_pt0_prob_emit_at_indel' )

# pickles
out_dict = {'lambda': lam,
            'mu': mu,
            'r_extend': r[None]}
with open(f'{neural_dir}/out_arrs/PARAMS-DICT_test-set_pt0_tkf92_indel_params.pkl','wb') as g:
    pickle.dump(out_dict, g)

# ascii files
maybe_write_matrix_to_ascii( out_folder = f'{neural_dir}/out_arrs',
                             mat = equl,
                             key = 'test-set_pt0_prob_emit_at_indel' )

offset = 1 - (lam/mu)
mean_indel_len = 1 / (1 - r)

with open(f'{neural_dir}/out_arrs/ASCII_test-set_pt0_tkf92_indel_params.txt', 'w') as g:
    g.write(f'insert rate, lambda: {lam}\n')
    g.write(f'deletion rate, mu: {mu}\n')
    g.write(f'offset: {offset}\n\n')
    
    g.write(f'extension prob, r: {r}\n')
    g.write(f'mean indel length: {mean_indel_len}\n')
    
    
