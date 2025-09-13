#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 13:29:16 2025

@author: annabel
"""
import os
import pandas as pd
import pickle

def read_loss_file(d, prefix):
    # get metadata from argparse
    path = f'{d}/model_ckpts/TRAINING_ARGPARSE.pkl'
    
    with open(path, 'rb') as f:
        args = pickle.load(f)
    
    anc_model = args.anc_model_type
    desc_model = args.desc_model_type
    assert anc_model == desc_model
    
    prediction_head = args.pred_model_type
    
    if prediction_head == 'neural_hmm':
        pred_config = args.pred_config
        global_or_local_dict = pred_config['global_or_local']
        
        subst_model_type = 'local ' + pred_config.get('subst_model_type', 'f81')
        
        if global_or_local_dict['tkf92_frag_size'] == 'local':
            indel_model_type = f'local tkf92'
        elif global_or_local_dict['tkf92_frag_size'] == 'global':
            indel_model_type = f'GLOBAL tkf92'
    
    elif prediction_head == 'feedforward':
        subst_model_type = 'nan'
        indel_model_type = 'nan'
        
    
    # read loss file
    path = f'{d}/logfiles/{prefix.upper()}-AVE-LOSSES.tsv'
    try: 
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = ['key', 'value']
        out_dict = pd.Series(df.value.values,index=df.key).to_dict()
        
        # add metadata to out dict
        model_type = f'{prediction_head}: {anc_model}'
        out_dict['type'] = model_type
        out_dict['subst_model_type'] = subst_model_type
        out_dict['indel_model_type'] = indel_model_type
        return out_dict

    except:
        return {}


def proc_all(fold, dset):
    ### all losses
    losses = [ read_loss_file(f'{fold}/{d}', dset) for d in os.listdir(fold) if d.startswith('RESULTS') ]
    losses = pd.DataFrame( losses )
    losses = losses.rename(columns={"sum_cond_logprobs": "sum_cond_loglikes"})
    losses = losses.dropna(axis=0)
    col_order = ['RUN',
                 'type',
                 'subst_model_type',
                 'indel_model_type',
                 'sum_cond_loglikes', 
                 'cond_ave_loss', 
                 'cond_ave_loss_seqlen_normed', 
                 'cond_ece']
    losses = losses[col_order]
    losses = losses.sort_values(by='sum_cond_loglikes', ascending=True)
    
    # save
    losses.to_csv(f'{fold}/ALL_{dset.upper()}_LOGLIKES_neural.tsv', sep='\t', index=False)
    

if __name__ == '__main__':
    proc_all(fold='.',  dset='test')
    
    
    
