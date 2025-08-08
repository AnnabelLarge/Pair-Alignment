#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 18:38:33 2025

@author: annabel_large
"""
import pandas as pd
import numpy as np

training_dir = 'RESULTS_train_indp-sites'
eval_dir = 'RESULTS_eval_frag-mix'

def read_loss_file(d):
    file = f'{d}/logfiles/test-set_pt0_FINAL-LOGLIKES.tsv'
    df = pd.read_csv(file, sep='\t', index_col=0)
    df['ID'] = df['ancestor'] + '///' + df['descendant']
    return df

df_train = read_loss_file(training_dir)
df_eval = read_loss_file(eval_dir)

# make sure rows are in the same order
assert (df_train['ID'] == df_eval['ID']).all()

# check losses
assert np.allclose(df_train['joint_logP'], df_eval['joint_logP'])
assert np.allclose(df_train['cond_logP'], df_eval['cond_logP'])
assert np.allclose(df_train['anc_logP'],   df_eval['anc_logP'])
assert np.allclose(df_train['desc_logP'],  df_eval['desc_logP'])

print('Loglikes match exactly!')
