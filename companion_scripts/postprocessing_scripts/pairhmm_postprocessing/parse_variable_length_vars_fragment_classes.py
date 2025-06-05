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

def main(fold):
    ### read folders
    all_runs = pd.read_csv(f'{fold}/ALL_test-set_joint_loglikes.tsv', 
                       sep='\t', 
                       index_col=0)
    
    
    ### record params
    params= {'run': [],
             'class_prob': [],
             'rate_mult': [],
             'lam': [],
             'mu': [],
             'r': [],
             'mean_fragment_len': [],
             'adjusted_lam': [],
             'adjusted_mu': []
             }
    
    def extract_info(sub_df, standard_model):
        for i in range(len(sub_df)):
            row = sub_df.iloc[i]
            d = row['run']
            indel_model = row['indel_model']
            params['run'].append(d)
                
            ### tkf params
            path = f'{fold}/{d}/out_arrs/PARAMS_TKF92_indel_params.txt'
            with open(path,'r') as f:
                for line in f:
                    pname, entry = line.strip().split(': ')
                    
                    if pname == 'insert rate, lambda':
                        lam = np.array([float(entry)])
                        params['lam'].append( lam )
                    
                    elif pname == 'deletion rate, mu':
                        mu = np.array([float(entry)])
                        params['mu'].append( mu )
                    
                    elif pname == 'extension prob, r':
                        r_vec = np.array( entry.split('\t') ).astype(float)
                        mean_frag_len = 1 / (1 - r_vec)
                        adj_lam = lam / mean_frag_len
                        adj_mu = mu / mean_frag_len
    
                        params['r'].append(r_vec) 
                        params['mean_fragment_len'].append( mean_frag_len )
                        params['adjusted_lam'].append( adj_lam )
                        params['adjusted_mu'].append( adj_mu )
    
            
            if indel_model == 'tkf91':
                params['r'].append( np.array([0]) )
                params['mean_fragment_len'].append( np.array([1]) )
                params['adjusted_lam'] = lam
                params['adjusted_mu'] = mu
            
            del path, line, pname, entry
            
            
            ### class probs, rate multipliers
            if standard_model:
                params['class_prob'].append( np.array([1]) )
                params['rate_mult'].append( np.array([1]) )
            
            elif not standard_model:
                # class probs
                path = f'{fold}/{d}/out_arrs/PARAMS_class_probs.txt'
                with open(path,'r') as f:
                    mat = np.array( [float(line.strip()) for line in f] )
                params['class_prob'].append(mat)
                del path, mat
                
                # rate multipliers
                path = f'{fold}/{d}/out_arrs/PARAMS_rate_multipliers.txt'
                with open(path,'r') as f:
                    mat = np.array( [float(line.strip()) for line in f] )
                params['rate_mult'].append(mat)
                del path, mat
                
    extract_info( all_runs[all_runs['aggby'].str.endswith('1_1')],standard_model=True )
    extract_info( all_runs[~all_runs['aggby'].str.endswith('1_1')],standard_model=False )
    
    # all variable-length things: class prob, rate multiplier, r, 
    #   adjusted lam, adjusted mu
    for i in range( len(params['run']) ):
        run = params['run'][i]
        class_prob = params['class_prob'][i]
        rate_mult = params['rate_mult'][i]
        r = params['r'][i]
        mean_fragment_len = params['mean_fragment_len'][i]
        adjusted_lam = params['adjusted_lam'][i]
        adjusted_mu = params['adjusted_mu'][i]
        
        final_mat = np.stack( [class_prob,
                               rate_mult,
                               r,
                               mean_fragment_len,
                               adjusted_lam,
                               adjusted_mu
                               ] )
        
        columns = [f'Class{i}' for i in range(r.shape[0])]
        indexes = ['class_prob',
                   'rate_mult',
                   'r',
                   'mean_fragment_len',
                   'adjusted_lam',
                   'adjusted_mu'
                   ]
        
        df = pd.DataFrame(final_mat, 
                          columns=columns,
                          index=indexes)
        
        new_path = f'{fold}/{run}/out_arrs/PARAMS_variable_per_class.tsv'
        df.to_csv(new_path, sep='\t')
        
        
if __name__ == '__main__':
    import sys
    
    # fold = sys.argv[1]
    
    fold = 'example_results_fragment_classes'
    
    main(fold)
        

