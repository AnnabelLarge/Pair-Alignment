#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:51:39 2025

@author: annabel
"""
import numpy as np
import pandas as pd
import pickle

neural_dir = 'RESULTS_train_neuralTKF_sync-params'

def at_most_two_types_exact(x):
    x_flat = x.reshape(x.shape[0], x.shape[1], -1) 
    uniq_counts = np.array([np.unique(x_flat[t], axis=0).shape[0] for t in range(x.shape[0])])
    ok = uniq_counts <= 2
    assert ok.all()
    
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

# should only be two sets of parameters: one at real sites, and one at padding sites
assert np.unique(lam).shape == (2,)
assert np.unique(mu).shape == (2,)
assert np.unique(r).shape == (2,)
at_most_two_types_exact(cond_prob_transit)

# remove B and L dims
lam = np.squeeze( np.array( param_dict['INDEL_MODEL_PARAMS']['lambda'] ) )[0,0]
mu = np.squeeze( np.array( param_dict['INDEL_MODEL_PARAMS']['mu'] ) )[0,0]
r = np.squeeze( np.array( param_dict['INDEL_MODEL_PARAMS']['r_extend'] ) )[0,0]
cond_prob_transit = np.exp( np.squeeze( np.array(param_dict['LOGPROB_TRANSITS']) ) )[:,0,...]


### emissions: for one f81, it's just equilibrium distribution
# equilibrium distribution
equl = np.exp( np.squeeze( np.array( param_dict['LOGPROB_EMIT_INDEL'] ) ) )
cond_prob_emit_match = np.exp( np.squeeze( np.array( param_dict['LOGPROB_EMIT_MATCH'] ) ) )

at_most_two_types_exact(cond_prob_emit_match)
at_most_two_types_exact(equl)

equl = np.exp( np.squeeze( np.array( param_dict['LOGPROB_EMIT_INDEL'] ) ) )[0,0,...]
cond_prob_emit_match = np.exp( np.squeeze( np.array( param_dict['LOGPROB_EMIT_MATCH'] ) ) )[:,0,...]


### write to files that are compatible with pairHMM code
# numpy arrays
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
    
    

