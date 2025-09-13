#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:39:06 2025

@author: annabel
"""
import pickle
import jax
from jax import numpy as jnp
import os
import pandas as pd
import numpy as np

def flatten_dict(d, parent_key="", sep="."):
    """
    Recursively flattens a nested dictionary.
    
    Example:
        {"a": {"b": 1, "c": {"d": 2}}, "e": 3}
    → {"a.b": 1, "a.c.d": 2, "e": 3}
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def count_params(file):
    # load trainstate file
    # file = f'{d}/model_ckpts/FINAL_PRED_BEST.pkl'
    with open(file,'rb') as f:
        param_dict = pickle.load(f)['params']['params']
    
    # flatten and count
    param_dict = flatten_dict( param_dict )
    param_count = 0
    for mat in param_dict.values():
        param_count += mat.size
    return param_count

def count_params_across_model(d):
    anc_enc_params = count_params(f'{d}/model_ckpts/ANC_ENC_BEST.pkl')
    desc_dec_params = count_params(f'{d}/model_ckpts/DESC_DEC_BEST.pkl')
    out_proj_params = count_params(f'{d}/model_ckpts/FINAL_PRED_BEST.pkl')
    
    out = {'anc_enc_params': anc_enc_params,
           'desc_dec_params': desc_dec_params,
           'out_proj_params': out_proj_params}
    
    return out

def extract_real_time(d):
    file = f'{d}/logfiles/TIMING.txt'

    if os.path.exists(file):
        with open(file,'r') as f:
            contents = [line.strip() for line in f]
        return float(contents[4].split('\t')[1])
    
    else:
        return jnp.nan
    
def extract_loglikes(d, dset):
    file = f'{d}/logfiles/{dset.upper()}-AVE-LOSSES.tsv'
    
    if os.path.exists(file):
        df = pd.read_csv(file, sep='\t', header=None, index_col=0).T
        df = df.rename(columns={"sum_cond_logprobs": "sum_cond_loglikes"})
        sum_cond_loglikes = df['sum_cond_loglikes'].item()
        cond_ece = df['cond_ece'].item()
        
    else:
        sum_cond_loglikes = jnp.nan
        cond_ece = jnp.nan
    
    return {f'{dset}_sum_cond_loglikes': sum_cond_loglikes,
            f'{dset}_cond_ece': cond_ece}


def gather_param_count_and_times(d):
    ### get metadata
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
    
    out_dict = {}
    out_dict['RUN'] = d
    model_type = f'{prediction_head}: {anc_model}'
    out_dict['type'] = model_type
    out_dict['subst_model_type'] = subst_model_type
    out_dict['indel_model_type'] = indel_model_type
    
    
    ### add other info
    test_loglikes = extract_loglikes(d, 'test')
    out_dict = {**out_dict, **test_loglikes}
    out_dict['ave_epoch_real_time'] = extract_real_time(d)
    
    param_count_dict = count_params_across_model(d)
    out_dict['num_parameters'] = sum( param_count_dict.values() )
    out_dict['anc_seq_embedder_params'] = param_count_dict['anc_enc_params']
    out_dict['desc_seq_embedder_params'] = param_count_dict['desc_dec_params']
    out_dict['prediction_head_params'] = param_count_dict['out_proj_params']
    return out_dict

def proc_all(fold):
    all_df = [ gather_param_count_and_times(f'{fold}/{d}') for d in os.listdir(fold) if d.startswith('RESULTS') ]
    all_df = pd.DataFrame( all_df )
    all_df = all_df.dropna(axis=0)
    
    col_order = ['RUN',
                 'type',
                 'subst_model_type',
                 'indel_model_type',
                 'ave_epoch_real_time',
                 'num_parameters',
                 'anc_seq_embedder_params',
                 'desc_seq_embedder_params',
                 'prediction_head_params',
                 'test_sum_cond_loglikes',
                 'test_cond_ece']
    all_df = all_df[col_order]
    
    # save
    all_df.to_csv(f'{fold}/ALL_TIME_PARAM-COUNTS_neural.tsv', sep='\t', index=False)
    

if __name__ == '__main__':
    proc_all(fold='.')
    
