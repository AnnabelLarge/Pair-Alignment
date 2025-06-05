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
    
    
    ### record params
    params= {'run': [],
             'class_prob': [],
             'rate_mult': [],
             'exch': [],
             'equl': [],
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
                
            ### exchangeabilities
            path = f'{fold}/{d}/out_arrs/PARAMS_exchangeabilities.npy'
            vec = np.load(path)
            params['exch'].append(vec)
            del path, vec
            
            
            ### equilibriums
            if standard_model:
                if t_per_samp:
                    path = f'{fold}/{d}/out_arrs/test-set_pt0_prob_emit_at_indel.npy'
                elif not t_per_samp:
                    path = f'{fold}/{d}/out_arrs/_prob_emit_at_indel.npy'
                    
            elif not standard_model:
                path = f'{fold}/{d}/out_arrs/PARAMS-ARR_equilibriums.npy'
                
            mat = np.load(path)
            
            if standard_model:
                mat = mat[None,:]
            
            params['equl'].append(mat)
            del path, mat
            
            
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
                
    extract_info( best[best['aggby'] == '1_1'],standard_model=True )
    extract_info( best[best['aggby'] != '1_1'],standard_model=False )
    
    
    ### aggregate and output
    # exchangeabilities, as (190, num_runs) table
    with open(f'pairhmm_postprocessing/exch_upper_tri_labels.txt','r') as f:
        indexes = [line.strip() for line in f]
    
    with open(f'pairhmm_postprocessing/LG08_exchangeability_vec.npy','rb') as f:
        lg08_ref = np.load(f)
    
    final_mat = np.stack( [lg08_ref] + params['exch'], 
                          axis=-1 )
    
    df = pd.DataFrame(final_mat, 
                      columns=['LG08']+params['run'],
                      index=indexes)
    
    df.to_csv(f'{fold}/BEST_exchangeabilities.tsv', sep='\t')
    
    del indexes, lg08_ref, final_mat, df
    
    
    # equilibrium distributions, as (20, num_classes_across_runs) table
    # also add class prob at the top
    indexes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 
               'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    columns = []
    for i in range( len(params['run']) ):
        runname = params['run'][i]
        class_probs = params['class_prob'][i]
        for c in range(class_probs.shape[0]):
            rounded_val = round( class_probs[c], 4 )
            
            tup = (runname,
                   f'Class{c}: prob={rounded_val}')
            
            columns.append(tup)
            
    final_mat = np.concatenate( params['equl'], axis=0 ).T
    
    df = pd.DataFrame(final_mat, 
                      columns=columns,
                      index=indexes)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv(f'{fold}/BEST_equilibriums.tsv', sep='\t')
    del indexes, columns, i, runname, class_probs, c, rounded_val, 
    del final_mat, df
    
    
    # raw lambda, mu as (2, num_runs) table
    lam = np.concatenate( params['lam'] )
    mu = np.concatenate( params['mu'] )
    final_mat = np.stack([lam, mu])
    
    df = pd.DataFrame(final_mat, 
                      columns=params['run'],
                      index=['raw_insert_rate_lambda', 'raw_delete_rate_mu'])
    
    df.to_csv(f'{fold}/BEST_raw_lambda_mu.tsv', sep='\t')
    
    
    
    # all variable-length things: class prob, rate multiplier, r, 
    #   adjusted lam, adjusted mu; flat text file, because idk how else to do it
    with open(f'{fold}/BEST_variable_len_params.tsv', 'w') as g:
        for i in range( len(params['run']) ):
            run = params['run'][i]
            class_prob = params['class_prob'][i]
            rate_mult = params['rate_mult'][i]
            r = params['r'][i]
            mean_fragment_len = params['mean_fragment_len'][i]
            adjusted_lam = params['adjusted_lam'][i]
            adjusted_mu = params['adjusted_mu'][i]
            
            g.write(run + '\n')
            g.write('='*(len(run)+2) + '\n')
            
            
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
            
            df.to_csv(g, sep='\t')
            
            g.write('\n')
            g.write('\n')



    
        
if __name__ == '__main__':
    import sys
    
    # fold = sys.argv[1]
    
    fold = 'example_results_fragment_classes'

    
    main(fold)
        

