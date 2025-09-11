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
        
    # read loss file
    path = f'{d}/logfiles/{prefix.upper()}-AVE-LOSSES.tsv'
    try: 
        df = pd.read_csv(path, sep='\t', header=None)
        df.columns = ['key', 'value']
        out_dict = pd.Series(df.value.values,index=df.key).to_dict()
        
        # add metadata to out dict
        out_dict['seed'] = seed
        out_dict['type'] = model_type
        out_dict['subst_model_type'] = subst_model_type
        out_dict['indel_model_type'] = indel_model_type
        out_dict['num_domain_mixtures'] = num_domain_mixtures
        out_dict['num_fragment_mixtures'] = num_fragment_mixtures
        out_dict['num_site_mixtures'] = num_site_mixtures
        out_dict['k_rate_mults'] = k_rate_mults
        
        return out_dict

    except:
        return {}


def proc_all(fold, dset):
    ### all losses
    losses = [ read_loss_file(f'{fold}/{d}', dset) for d in os.listdir(fold) if d.startswith('RESULTS') ]
    losses = pd.DataFrame( losses )
    losses = losses.dropna(axis=0)
    
    col_order = ['RUN',
                 'seed',
                 'type',
                 'subst_model_type',
                 'indel_model_type',
                 'num_domain_mixtures',
                 'num_fragment_mixtures',
                 'num_site_mixtures',
                 'k_rate_mults',
                 'sum_joint_loglikes',
                 'joint_ave_loss',
                 'joint_ave_loss_seqlen_normed',
                 'joint_ece', 
                 'sum_cond_loglikes', 
                 'cond_ave_loss', 
                 'cond_ave_loss_seqlen_normed', 
                 'cond_ece',  
                 'sum_anc_loglikes', 
                 'anc_ave_loss', 
                 'anc_ave_loss_seqlen_normed', 
                 'anc_ece']
    losses = losses[col_order]
    losses = losses.sort_values(by='sum_cond_loglikes', ascending=True)
    
    # save
    losses.to_csv(f'{fold}/ALL_{dset.upper()}_LOGLIKES_pairhmms.tsv', sep='\t', index=False)
    
    
    ### best of triplicates
    losses['aggby'] = losses['RUN'].str.replace(r"_seed\d+", "", regex=True)
    best = losses.loc[losses.groupby("aggby")["sum_joint_loglikes"].idxmin()]
    assert len( set(losses['aggby']) - set(best['aggby']) ) == 0

    best = best.drop('aggby', axis=1)
    best = best.drop('seed', axis=1)
    best = best.sort_values(by='sum_cond_loglikes', ascending=True)
    best.to_csv( f'{fold}/BEST_{dset.upper()}_LOGLIKES_pairhmms.tsv', sep='\t' )


if __name__ == '__main__':
    proc_all(fold='.',  dset='test')
    
    
    
