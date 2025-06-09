#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 17:47:18 2025

@author: annabel
"""
import numpy as np
import pandas as pd
import sys
import os

def main(fold, t_per_samp):
    ### read folders
    best = pd.read_csv(f'{fold}/BEST_test-set_joint_loglikes.tsv', 
                       sep='\t', 
                       index_col=0)
    best['aggby'] = best['num_site_classes'].astype(str) + '_' + best['num_fragment_classes'].astype(str)
    
    columns = []
    scaled_rate_matrices = []
    for i in range(len(best)):
        row = best.iloc[i]
        d = row['run']
        indel_model = row['indel_model']
        num_mixtures = row['num_site_classes']
        
        for c in range(num_mixtures):
            if t_per_samp:
                path = f'{fold}/{d}/out_arrs/test-set_pt0_class-{c}_rate_matrix_times_rho.npy'
            elif not t_per_samp:
                path = f'{fold}/{d}/out_arrs/_class-{c}_rate_matrix_times_rho.npy'
            
            with open(path,'rb') as f:
                final_mat = np.load(f)
            
            tup = (d, f'Class{c}')
            columns.append(tup)
            scaled_rate_matrices.append(final_mat)
        
            del c, path, tup
            
        del row, d, indel_model, num_mixtures, i
    
    
    aas_in_order = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 
                    'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    indexes = np.zeros( (20,20) ).astype(str)
    for i, aa_from in enumerate(aas_in_order):
        for j, aa_to in enumerate(aas_in_order):
            indexes[i,j] = (f'{aa_from}_{aa_to}')
    indexes = indexes.reshape(-1)
    scaled_rate_matrices = np.stack( scaled_rate_matrices, axis=0 )
    scaled_rate_matrices = scaled_rate_matrices.reshape( len(scaled_rate_matrices), -1 ).T
    
    df = pd.DataFrame(scaled_rate_matrices, 
                      columns = columns,
                      index=indexes)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    
    df.to_csv(f'{fold}/BEST_scaled-rate-mats.tsv', sep='\t')


if __name__ == '__main__':
    import sys
    
    fold = sys.argv[1]
    t_per_samp = sys.argv[2]
    
    if t_per_samp.lower() in ['true', 't']:
        t_per_samp = True
    
    elif t_per_samp.lower() in ['false', 'f']:
        t_per_samp = False
    
    
    # fold = 'example_results_indp_sites'
    # t_per_samp = True
    
    main(fold, t_per_samp)
        

