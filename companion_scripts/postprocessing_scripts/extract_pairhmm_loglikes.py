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
    
    seed = args.rng_seednum
    pred_config = args.pred_config
    
    subst_model_type = pred_config['subst_model_type']
    indel_model_type = pred_config['indel_model_type']
    num_fragment_mixtures = pred_config['num_fragment_mixtures']
    num_site_mixtures = pred_config['num_site_mixtures']
    k_rate_mults = pred_config['k_rate_mults']
    
    # read loss file
    path = f'{d}/logfiles/{prefix.upper()}-AVE-LOSSES.tsv'
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['key', 'value']
    out_dict = pd.Series(df.value.values,index=df.key).to_dict()
    
    # add metadata to out dict
    out_dict['seed'] = seed
    out_dict['subst_model_type'] = subst_model_type
    out_dict['indel_model_type'] = indel_model_type
    out_dict['num_fragment_mixtures'] = num_fragment_mixtures
    out_dict['num_site_mixtures'] = num_site_mixtures
    out_dict['k_rate_mults'] = k_rate_mults
    
    return out_dict

def proc_all(fold, dset):
    losses = pd.DataFrame([read_loss_file(f'{fold}/{d}', dset) for d in os.listdir(fold) 
                           if d.startswith('RESULTS')])
    
    col_order = ['RUN',
                 'seed',
                 'subst_model_type',
                 'indel_model_type',
                 'num_fragment_mixtures',
                 'num_site_mixtures',
                 'k_rate_mults',
                 'sum_joint_loglikes',
                 'joint_ave_loss',
                 'joint_ave_loss_seqlen_normed',
                 'joint_ece', 
                 'joint_perplexity', 
                 'sum_cond_loglikes', 
                 'cond_ave_loss', 
                 'cond_ave_loss_seqlen_normed', 
                 'cond_ece', 
                 'cond_perplexity', 
                 'sum_anc_loglikes', 
                 'anc_ave_loss', 
                 'anc_ave_loss_seqlen_normed', 
                 'anc_ece', 
                 'anc_perplexity', 
                 'sum_desc_loglikes', 
                 'desc_ave_loss', 
                 'desc_ave_loss_seqlen_normed', 
                 'desc_ece', 
                 'desc_perplexity']
    losses = losses[col_order]
    
    losses = losses.sort_values(by='sum_joint_loglikes')
    
    losses.to_csv(f'{fold}/ALL_{dset.upper()}_LOGLIKES.tsv', sep='\t', index=False)


if __name__ == '__main__':
    proc_all(fold='.', 
             dset='train')
    
    
    
