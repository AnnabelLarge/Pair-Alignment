#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:33:48 2025

@author: annabel
"""
import numpy as np
import pandas as pd

# threshold for this is lower due to training differences
THRESHOLD = 1e-3

times = ['marg-over-times', 't-per-samp']
subs = ['f81', 'gtr']
indels = ['tkf91', 'tkf92']

test_dir = 'tests/neural_hmm/no_seq_embs_vs_simple_site_class'

for t_str in times:
    for s_str in subs:
        for i_str in indels:
            true_dir = f'{test_dir}/pairhmm_reference/TRUE_{t_str}_{s_str}_{i_str}'
            pred_dir = f'{test_dir}/neuralTKF_train/RESULTS_{t_str}_{s_str}_{i_str}'
            
            true_loglikes_file = f'{true_dir}/logfiles/test-set_pt0_FINAL-LOGLIKES.tsv'
            pred_loglikes_file = f'{pred_dir}/logfiles/test-set_pt0_FINAL-LOGLIKES.tsv'
            
            true_loglikes_df = pd.read_csv(true_loglikes_file, sep='\t', index_col=0)
            pred_loglikes_df = pd.read_csv(pred_loglikes_file, sep='\t', index_col=0)
            
            assert np.allclose( true_loglikes_df['dataloader_idx'], 
                                pred_loglikes_df['dataloader_idx'] )
            
            true_cond_loglike = true_loglikes_df['cond_logP'].to_numpy()
            pred_cond_loglike = pred_loglikes_df['logP'].to_numpy()
            
            assert np.allclose( true_cond_loglike,
                                pred_cond_loglike,
                                atol=0,
                                rtol=THRESHOLD )

print(f'Neural loglikes within {THRESHOLD*100}% of pairHMM loglikes')
