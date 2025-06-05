#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:28:28 2025

@author: annabel
"""
import pandas as pd
import os
import sys
import pickle
from tqdm import tqdm


def main(fold, dset_prefix):
    all_dirs = [d for d in os.listdir(fold) if d.startswith('RESULTS')]
    
    all_out = []
    for d in tqdm(all_dirs):
        # read total loglike
        path = f'{fold}/{d}/logfiles'
        all_result_files_in_d = [file for file in os.listdir(path) if file.startswith(dset_prefix) 
                                 and file.endswith('FINAL-LOGLIKES.tsv')]
        total_joint_nll = 0
        for file in all_result_files_in_d:
            df = pd.read_csv(f'{path}/{file}', sep='\t')
            total_joint_nll += df['joint_logP'].sum()
        
        del path, all_result_files_in_d, file, df
        
        
        # get metadata from training argparse and config table
        argparse_table_loc = f'{fold}/{d}/model_ckpts/CONFIG-TABLE.tsv'
        df = pd.read_csv(argparse_table_loc, index_col=0, sep='\t').T
        pred_model_type = df['pred_model_type'].item()
        
        del argparse_table_loc, df
        
        argparse_table_loc = f'{fold}/{d}/model_ckpts/PRED-CONFIG.tsv'  
        df = pd.read_csv(argparse_table_loc, index_col=0, sep='\t').T
        
        num_mixtures = df['num_mixtures'].item()
        
        del argparse_table_loc
        
        
        # fill in output table
        if pred_model_type == 'pairhmm_indp_sites':
            num_fragment_classes = 1
            indel_model = df['indel_model_type'].item()
        elif pred_model_type == 'pairhmm_frag_and_site_classes':
            num_fragment_classes = num_mixtures
            indel_model = 'tkf92'
        
        # aggregate with combo of indel_model type and number of classes
        agg_name = f'{indel_model}_{num_mixtures}_{num_fragment_classes}'
        
        all_out.append({'run': d,
                        'pred_model_type': pred_model_type,
                        'substitution_model': 'gtr',
                        'indel_model': indel_model,
                        'num_site_classes': num_mixtures,
                        'num_fragment_classes': num_fragment_classes,
                        'sum_joint_logprobs': total_joint_nll,
                        'aggby': agg_name})
    
    all_out = pd.DataFrame(all_out)
    all_out = all_out.sort_values(by='aggby')
    
    idx = all_out.groupby('aggby')['sum_joint_logprobs'].idxmin()
    best = all_out.loc[idx]
    best = best.drop('aggby', axis=1)
    
    # save
    all_out.to_csv(f'{fold}/ALL_{dset_prefix}_joint_loglikes.tsv', sep='\t')
    best.to_csv(f'{fold}/BEST_{dset_prefix}_joint_loglikes.tsv', sep='\t')
    

if __name__ == '__main__':
    import sys
    
    # fold = sys.argv[1]
    # dset_prefix = sys.argv[2]
        
    fold = 'example_results_fragment_classes'
    dset_prefix = 'test-set'

    main(fold, dset_prefix)