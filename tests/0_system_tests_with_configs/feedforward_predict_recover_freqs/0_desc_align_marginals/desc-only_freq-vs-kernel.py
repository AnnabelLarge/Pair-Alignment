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

results_dir = 'RESULTS_recov-desc-freq'
prefix = 'twoSamp'


### load the kernel and softmax-transform it
params_file = f'{results_dir}/model_ckpts/FINAL_PRED_BEST.pkl'
with open(params_file,'rb') as f:
    params = pickle.load(f)

kernel = np.array( params['params']['params']['FEEDFORWARD PREDICT/final projection']['kernel'] )
print(f'raw kernel size: {kernel.shape}')

# softmax transform,
kernel = np.squeeze(kernel)
softm_kernel = np.array( jax.nn.softmax(kernel) )

### load the calculate frequency vector
with open(f'{prefix}_desc-align_counts.npy','rb') as f:
    desc_counts = np.load(f)
print(f'raw counts matrix size: {desc_counts.shape}')

# transform to freqs
desc_freq = desc_counts / desc_counts.sum()

# remove tokens unseen by the output network: <bos>, <pad>
desc_freq = desc_freq[2:]
softm_kernel = softm_kernel[2:]

# get differences
abs_diff = np.abs( desc_freq - softm_kernel )
relative_diff = np.where( desc_freq != 0,
                          abs_diff / desc_freq,
                          0 )

### output dataframe
see = np.stack([desc_freq, softm_kernel, abs_diff, relative_diff]).T
df = pd.DataFrame(see, columns = ['true freq', 'softmaxed kernel', 'abs diff', 'rel diff'])
df.to_csv(f'DESC-ONLY_freq-vs-kernel_{results_dir}.tsv', sep='\t')