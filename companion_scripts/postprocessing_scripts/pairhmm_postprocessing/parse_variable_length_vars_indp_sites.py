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
             'rate_mult': []
             }
    
    def extract_info(sub_df, standard_model):
        for i in range(len(sub_df)):
            row = sub_df.iloc[i]
            d = row['run']
            indel_model = row['indel_model']
            params['run'].append(d)
            
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
    
    
    ### aggregate and output all variable-length things: class prob, rate multiplier
    for i in range( len(params['run']) ):
        run = params['run'][i]
        
        class_prob = params['class_prob'][i]
        rate_mult = params['rate_mult'][i]
        
        final_mat = np.stack( [class_prob,
                               rate_mult
                               ] )
        
        columns = [f'Class{i}' for i in range(class_prob.shape[0])]
        indexes = ['class_prob',
                   'rate_mult'
                   ]
        
        df = pd.DataFrame(final_mat, 
                          columns=columns,
                          index=indexes)
        
        new_path = f'{fold}/{run}/out_arrs/PARAMS_variable_per_class.tsv'
        df.to_csv(new_path, sep='\t')
        
if __name__ == '__main__':
    import sys
    
    fold = sys.argv[1]
    
    # fold = 'example_results_indp_sites'
    
    main(fold)
        

