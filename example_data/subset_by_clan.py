#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 19:00:02 2025

@author: annabel

pull some small sample datasets from OOD valid
"""
import pandas as pd
import numpy as np


def read_mat(file):
    with open(file,'rb') as f:
        return np.load(f)

def subset_mat(file, idx_to_keep):
    mat = read_mat(file)
    return mat[idx_to_keep, ...]

def subset(meta_df, cl, num_samps, new_prefix):
    # subset metadata
    sub_df = meta_df[ meta_df['clan']==cl ].iloc[0:num_samps]
    
    assert len(sub_df) > 0, 'no samples!'
    
    sample_idx = list( sub_df.index )
    
    # subset major numpy matrices
    out = {'aligned_mats': subset_mat(f'FAMCLAN-CHERRIES_{prefix}_aligned_mats.npy', sample_idx),
           'delCounts': subset_mat(f'FAMCLAN-CHERRIES_{prefix}_delCounts.npy', sample_idx),
           'insCounts': subset_mat(f'FAMCLAN-CHERRIES_{prefix}_insCounts.npy', sample_idx),
           'seqs_unaligned': subset_mat(f'FAMCLAN-CHERRIES_{prefix}_seqs_unaligned.npy', sample_idx),
           'subCounts': subset_mat(f'FAMCLAN-CHERRIES_{prefix}_subCounts.npy', sample_idx),
           'transCounts_five_by_five': subset_mat(f'FAMCLAN-CHERRIES_{prefix}_transCounts_five_by_five.npy', sample_idx),
           'transCounts': subset_mat(f'FAMCLAN-CHERRIES_{prefix}_transCounts.npy', sample_idx) 
           }
    
    # AAcounts; save this to two files
    _, counts= np.unique( out['aligned_mats'][...,[0,1]], return_counts=True )
    out['AAcounts'] = counts[3:23]
    
    # times
    times = pd.read_csv(f'FAMCLAN-CHERRIES_{prefix}_pair-times.tsv', 
                        sep='\t',
                        header=None).iloc[0:num_samps]
    
    # save all
    for suff, val in out.items():
        with open(f'{new_prefix}_{suff}.npy','wb') as g:
            np.save(g, val)
    
    with open(f'{new_prefix}_AAcounts_subsOnly.npy','wb') as g:
        np.save(g, out['AAcounts'])
    
    times.to_csv(f'{new_prefix}_pair-times.tsv',
                 sep='\t',
                 header=None,
                 index=None)
    
    sub_df.to_csv(f'{new_prefix}_metadata.tsv',
                  sep='\t')
            

if __name__ == '__main__':
    prefix = 'OOD_valid'
    meta_df = pd.read_csv(f'FAMCLAN-CHERRIES_{prefix}_metadata.tsv', 
                          sep='\t',
                          index_col=0)
    
    # CL0376: extract 40
    subset(meta_df = meta_df, 
           cl = 'CL0376', 
           num_samps = 40, 
           new_prefix = 'fortySamp')
    
    # CL0722: extract 8
    subset(meta_df = meta_df, 
           cl = 'CL0722', 
           num_samps = 8, 
           new_prefix = 'eightSamp')
    
    # CL0734: extract 2
    subset(meta_df = meta_df, 
           cl = 'CL0734', 
           num_samps = 2, 
           new_prefix = 'twoSamp')
    
