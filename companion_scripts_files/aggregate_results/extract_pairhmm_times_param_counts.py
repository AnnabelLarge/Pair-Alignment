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

def count_params(d):
    # load trainstate file
    file = f'{d}/model_ckpts/FINAL_PRED_BEST.pkl'
    with open(file,'rb') as f:
        param_dict = pickle.load(f)['params']['params']
    
    # flatten and count
    param_dict = flatten_dict( param_dict )
    param_count = 0
    for mat in param_dict.values():
        param_count += mat.size
    return param_count

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
        sum_joint_loglikes = df['sum_joint_loglikes'].item()
        joint_ece = df['joint_ece'].item()
        cond_ece = df['cond_ece'].item()
        
    else:
        sum_joint_loglikes = jnp.nan
        joint_ece = jnp.nan
        cond_ece = jnp.nan
    
    return {f'{dset}_sum_joint_loglikes': sum_joint_loglikes,
            f'{dset}_joint_ece': joint_ece,
            f'{dset}_cond_ece': cond_ece}


def gather_param_count_and_times(d):
    ### get metadata
    path = f'{d}/model_ckpts/TRAINING_ARGPARSE.pkl'
    
    with open(path, 'rb') as f:
        args = pickle.load(f)
    
    seed = args.rng_seednum
    pred_config = args.pred_config
    
    subst_model_type = pred_config['subst_model_type']
    indel_model_type = pred_config.get('indel_model_type', 'tkf92')
    num_domain_mixtures = pred_config.get('num_domain_mixtures', 1)
    num_fragment_mixtures = pred_config['num_fragment_mixtures']
    num_site_mixtures = pred_config['num_site_mixtures']
    k_rate_mults = pred_config['k_rate_mults']
    
    # figure out pairhmm type
    key = (num_domain_mixtures > 1,
           num_fragment_mixtures > 1,
           num_site_mixtures > 1)

    model_type_map = {
        (True,  True,  True):  "domain mix pairhmm",
        (False, True,  True):  "fragment mix pairhmm",
        (False, False, True):  "site mix pairhmm",
        (False, False, False): "reference pairhmm",
    }
    
    model_type = model_type_map.get(key, "unsupported")
    
    out_dict = {}
    out_dict['RUN'] = d
    out_dict['seed'] = seed
    out_dict['type'] = model_type
    out_dict['subst_model_type'] = subst_model_type
    out_dict['indel_model_type'] = indel_model_type
    out_dict['num_domain_mixtures'] = num_domain_mixtures
    out_dict['num_fragment_mixtures'] = num_fragment_mixtures
    out_dict['num_site_mixtures'] = num_site_mixtures
    out_dict['k_rate_mults'] = k_rate_mults
    
    
    ### add other info
    train_loglikes = extract_loglikes(d, 'train')
    test_loglikes = extract_loglikes(d, 'test')
    out_dict = {**out_dict, **train_loglikes}
    out_dict = {**out_dict, **test_loglikes}
    out_dict['ave_epoch_real_time'] = extract_real_time(d)
    out_dict['num_parameters'] = count_params(d)
    return out_dict

def proc_all(fold):
    all_df = [ gather_param_count_and_times(f'{fold}/{d}') for d in os.listdir(fold) if d.startswith('RESULTS') ]
    all_df = pd.DataFrame( all_df )
    all_df = all_df.dropna(axis=0)
    
    col_order = ['RUN',
                 'seed',
                 'type',
                 'subst_model_type',
                 'indel_model_type',
                 'num_domain_mixtures',
                 'num_fragment_mixtures',
                 'num_site_mixtures',
                 'k_rate_mults',
                 'ave_epoch_real_time',
                 'num_parameters',
                 'train_sum_joint_loglikes',
                 'test_sum_joint_loglikes',
                 'test_joint_ece',
                 'test_cond_ece']
    all_df = all_df[col_order]
    
    # save
    all_df.to_csv(f'{fold}/ALL_TIME_PARAM-COUNTS_pairhmms.tsv', sep='\t', index=False)
    
    
    ### best of triplicates
    all_df['aggby'] = all_df['RUN'].str.replace(r"_seed\d+", "", regex=True)
    best = all_df.loc[all_df.groupby("aggby")["test_sum_joint_loglikes"].idxmin()]
    assert len( set(all_df['aggby']) - set(best['aggby']) ) == 0

    best = best.drop('aggby', axis=1)
    best = best.drop('seed', axis=1)
    best = best.sort_values(by='test_sum_joint_loglikes', ascending=True)
    best.to_csv( f'{fold}/BEST_TIME_PARAM-COUNTS_pairhmms.tsv', sep='\t' )
    

if __name__ == '__main__':
    proc_all(fold='.')
    
