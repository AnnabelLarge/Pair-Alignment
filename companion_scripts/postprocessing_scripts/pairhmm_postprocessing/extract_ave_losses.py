#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 17:47:18 2025

@author: annabel
"""
import pandas as pd
import sys
import os


def main(fold, dset_prefix, all_or_best):
    # read folders
    df = pd.read_csv(f'{fold}/{all_or_best.upper()}_{dset_prefix}_joint_loglikes.tsv', 
                       sep='\t', 
                       index_col=0)
    
    # get summary stats per run
    if dset_prefix == 'test-set':
        file = 'TEST-AVE-LOSSES.tsv'
    
    elif dset_prefix == 'train-set':
        file = 'TRAIN-AVE-LOSSES.tsv'

    elif dset_prefix == 'dset':
        file = 'AVE-LOSSES.tsv'
    
    else:
        raise NotImplementedError(f'no mapping for {dset_prefix}')
    
    all_loss_stats = []
    for d in df['run']:
        path = f'{fold}/{d}/logfiles/{file}'
        losses_df = pd.read_csv(path, sep='\t', index_col=0, header=None).T
        all_loss_stats.append(losses_df)
    all_loss_stats = pd.concat(all_loss_stats)
    all_loss_stats.rename(columns={'RUN': 'run'}, inplace=True)
    
    df = pd.merge(df, all_loss_stats, on='run')
    
    if 'aggby' in df.columns:
        df = df.drop('aggby', axis=1)

    df.to_csv(f'{fold}/{all_or_best.upper()}_AVE-LOSSES_{dset_prefix}.tsv', sep='\t')
    
        
if __name__ == '__main__':
    import sys
    
    # fold = sys.argv[1]
    # dset_prefix = sys.argv[2]
    
    fold = 'example_results_fragment_classes'
    dset_prefix = 'test-set'
    all_or_best = 'ALL'
    
    main(fold, dset_prefix, all_or_best)
        

