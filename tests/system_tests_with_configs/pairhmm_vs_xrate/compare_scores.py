#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 12:37:55 2025

@author: annabel
"""
import numpy as np
import pandas as pd

def load_tsv(file):
    return pd.read_csv(file, sep='\t', index_col=0)

def load_mat(file):
    with open(file,'rb') as f:
        return np.load(f)

df = load_tsv('tests/pairhmm_vs_xrate/get_xrate_loglikes/one-GTR_PF07734_score_per_sample_parsed.tsv')
xrate_joint_loglike_bits = pd.Series(df.inside_loglike_bits.values,index=df.pair).to_dict()
del df

scores_df = load_tsv('RESULTS/logfiles/withpairID_test-set_pt0_FINAL-LOGLIKES.tsv')[['pairID', 'joint_logP']]
scores_df['joint_logP'] = -scores_df['joint_logP']

scores_df.columns = ['pairID', 'my_code_nats']
scores_df['xrate_bits'] = scores_df['pairID'].apply(lambda x: xrate_joint_loglike_bits[x])
del xrate_joint_loglike_bits

scores_df['xrate_nats'] = scores_df['xrate_bits'] * np.log(2)
scores_df['my_code_bits'] = scores_df['my_code_nats'] / np.log(2)
bitscores = scores_df[['pairID', 'xrate_bits', 'my_code_bits']]
natscores = scores_df[['pairID', 'xrate_nats', 'my_code_nats']]
del scores_df

bitscores['abs_diff'] = np.abs( bitscores['xrate_bits'] - bitscores['my_code_bits'] )
bitscores['rel_diff'] = bitscores['abs_diff'] / np.abs( bitscores['xrate_bits'] )
print(f"In bits, maximum relative difference: {bitscores['rel_diff'].max()}")

natscores['abs_diff'] = np.abs( natscores['xrate_nats'] - natscores['my_code_nats'] )
natscores['rel_diff'] = natscores['abs_diff'] / np.abs( natscores['xrate_nats'] )
print(f"In nats, maximum relative difference: {natscores['rel_diff'].max()}")

bitscores.to_csv(f'bitscore_difference.tsv', sep='\t')
natscores.to_csv(f'natscore_difference.tsv', sep='\t')

