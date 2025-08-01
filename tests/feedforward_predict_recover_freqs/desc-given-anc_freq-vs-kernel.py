#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:36:59 2025

@author: annabel_large
"""
import pandas as pd
import numpy as np
import pickle
import jax

results_dir = 'RESULTS_desc-given-anc_recover-freq-only'
prefix = 'twoSamp'


### load the kernel and softmax-transform it
params_file = f'{results_dir}/model_ckpts/FINAL_PRED_BEST.pkl'
with open(params_file,'rb') as f:
    params = pickle.load(f)

kernel = np.array( params['params']['params']['FEEDFORWARD PREDICT/final projection']['kernel'] )
print(f'raw kernel size: {kernel.shape}')

# softmax transform
softm_kernel = np.array( jax.nn.softmax(kernel, axis=1) )


### load the calculate frequency vector
with open(f'{prefix}_desc-align_given_current_anc_counts.npy','rb') as f:
    desc_given_align_counts = np.load(f)
print(f'raw counts matrix size: {desc_given_align_counts.shape}')

# transform to freqs
desc_given_align_freq = desc_given_align_counts / ( desc_given_align_counts.sum(axis=1)[:,None] )
desc_given_align_freq = np.nan_to_num(x=desc_given_align_freq, nan=0)

# remove tokens unseen by the output network: <bos>, <pad>
desc_given_align_freq = desc_given_align_freq[2:,2:]
softm_kernel = softm_kernel[2:,2:]

# get differences
abs_diff = np.abs( desc_given_align_freq - softm_kernel )
relative_diff = np.where( desc_given_align_freq != 0,
                          abs_diff / desc_given_align_freq,
                          0 )

### output dataframe
rows, cols = np.indices(desc_given_align_freq.shape)
flat_indices = np.stack((rows.flatten(), cols.flatten()), axis=1)
desc_given_align_freq = desc_given_align_freq.flatten()
softm_kernel = softm_kernel.flatten()
abs_diff = abs_diff.flatten()
relative_diff = relative_diff.flatten()

see = np.stack([flat_indices[:,0],
                flat_indices[:,1],
                desc_given_align_freq, 
                softm_kernel,
                abs_diff, 
                relative_diff]).T
df = pd.DataFrame(see, columns = ['i','j','true freq', 'softmaxed kernel', 'abs diff', 'rel diff'])
df.to_csv(f'DESC-GIVEN-ANC_freq-vs-kernel_{results_dir}.tsv', sep='\t')