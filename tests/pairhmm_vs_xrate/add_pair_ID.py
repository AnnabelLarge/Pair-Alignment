#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 12:25:48 2025

@author: annabel
"""
import numpy as np
import pandas as pd


def load_tsv(file):
    return pd.read_csv(file, sep='\t', index_col=0)

def load_mat(file):
    with open(file,'rb') as f:
        return np.load(f)


meta_df = load_tsv('PF07734/PF07734_metadata.tsv')
meta_df['ID'] = meta_df['ancestor'] + '///' + meta_df['descendant']
meta_df = meta_df[['ID','pairID']]
id_to_pairname = pd.Series(meta_df.pairID.values,index=meta_df.ID).to_dict()

my_script_scores = load_tsv('RESULTS/logfiles/test-set_pt0_FINAL-LOGLIKES.tsv')
my_script_scores['ID'] = my_script_scores['ancestor'] + '///' + my_script_scores['descendant']
my_script_scores['pairID'] = my_script_scores['ID'].apply(lambda x: id_to_pairname[x])
my_script_scores = my_script_scores.drop('ID', axis=1)

my_script_scores.to_csv(f'RESULTS/logfiles/withpairID_test-set_pt0_FINAL-LOGLIKES.tsv', sep='\t')

