#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:49:11 2025

@author: annabel
"""
import pandas as pd


suffix = 'neural_tkf'

def read_df(file):
    return pd.read_csv(file, sep='\t', index_col=None)

pt1 = read_df(f'ALL_TEST_LOGLIKES_{suffix}.tsv')
pt2 = read_df(f'ALL_TIME_PARAM-COUNTS_{suffix}.tsv')

pt2['RUN'] = pt2['RUN'].str.replace('./','')
cols_to_keep = ['RUN','ave_epoch_real_time', 'num_parameters']
pt2 = pt2[cols_to_keep]

merged = pd.merge(pt1, pt2, on='RUN')

new_col_order = ['RUN', 
                 'type', 
                 'ave_epoch_real_time', 
                 'num_parameters', 
                 'subst_model_type', 
                 'indel_model_type',
                 'sum_cond_loglikes',
                 'cond_ave_loss', 
                 'cond_ave_loss_seqlen_normed', 
                 'cond_ece']
merged = merged[new_col_order]

merged.to_csv(f'MERGED_{suffix}.tsv', sep='\t')

