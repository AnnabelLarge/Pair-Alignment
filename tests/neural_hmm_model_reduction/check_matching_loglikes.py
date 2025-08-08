#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 18:38:33 2025

@author: annabel_large
"""
import pandas as pd
import numpy as np

# training_neural_dir = 'RESULTS_train_neuralTKF_sync-params'
# eval_pairhmm_dir = 'RESULTS_pairhmm_eval_sync-local-params'

training_neural_dir = 'RESULTS_train_neuralTKF_force-global'
eval_pairhmm_dir = 'RESULTS_pairhmm_eval_force-global'

def read_loss_file(d):
    file = f'{d}/logfiles/test-set_pt0_FINAL-LOGLIKES.tsv'
    df = pd.read_csv(file, sep='\t', index_col=0)
    df['ID'] = df['ancestor'] + '///' + df['descendant']
    return df

df_train = read_loss_file(training_neural_dir)
df_eval = read_loss_file(eval_pairhmm_dir)

assert (df_train['ID'] == df_eval['ID']).all()
assert np.allclose(df_train['logP'], df_eval['cond_logP'])
print('Loglikes match exactly!')
