#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:01:23 2025

@author: annabel
"""
import json
import os
import argparse
import jax
import pickle
import shutil
import gc

from dloaders.init_dataloader import init_dataloader
from utils.edit_argparse import (enforce_valid_defaults,
                                 fill_with_default_values,
                                 share_top_level_args)

def load_dset_pkl_fn(file_to_load, 
                     args,
                     collate_fn):
    print(f'loading pickle: {file_to_load}')
    
    with open(file_to_load,'rb') as f:
        dset_dict = pickle.load(f)
    
    # add dataloader objects
    test_dl = init_dataloader(args = args, 
                              shuffle = False,
                              pytorch_custom_dset = dset_dict['test_dset'],
                              collate_fn = collate_fn)
    dset_dict['test_dl'] = test_dl
    
    if 'training_dset' in dset_dict.keys():
        training_dl = init_dataloader(args = args, 
                                        shuffle = True,
                                        pytorch_custom_dset = dset_dict['training_dset'],
                                        collate_fn = collate_fn)
        dset_dict['training_dl'] = training_dl
    return dset_dict
        

def main(args, load_dset_pkl=None):
    pred_model_type = args.pred_model_type
    
    # import correct wrappers, dataloader initializers
    if pred_model_type == 'pairhmm_indp_sites':
        from dloaders.init_counts_dset import init_counts_dset as init_datasets
        from dloaders.CountsDset import jax_collator as collate_fn
        
    elif pred_model_type in ['pairhmm_frag_and_site_classes',
                             'neural_hmm',
                             'feedforward']:
        from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
        from dloaders.FullLenDset import jax_collator as collate_fn
            
    # make dataloder list
    if load_dset_pkl is None:
        dload_dict = init_datasets( args,
                                      'train',
                                      training_argparse = None,
                                      include_dataloader = True)
    else:
        dload_dict = load_dset_pkl_fn(args = args,
                                      file_to_load = load_dset_pkl,
                                      collate_fn = collate_fn)
    
    training_dset = dload_dict['training_dset']
    del dload_dict
    
    
    ### alter argparse object and save
    args.pred_config['training_dset_emit_counts'] = training_dset.emit_counts
    fill_with_default_values(args)
    enforce_valid_defaults(args)
    share_top_level_args(args)
    
    # missing best epoch, but oh well
    model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
    with open(f'{model_ckpts_dir}/TRAINING_ARGPARSE.pkl', 'wb') as g:
        pickle.dump(args, g)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-configs',
                        type = str,
                        required=True,
                        help='Load configs from file or folder of files, in json format.')
    
    parser.add_argument('-load_dset_pkl',
                        type = str,
                        default=None,
                        help='name of the pre-computed pytorch dataset pickle object')
    
    args = parser.parse_args()
    assert args.configs.endswith('.json'), "input is one JSON file"
    print(f'CONVERTING: {args.configs}')
    
    with open(args.configs, 'r') as f:
        contents = json.load(f)
    t_args = argparse.Namespace()
    t_args.__dict__.update(contents)
    args = parser.parse_args(namespace=t_args)
    del t_args, contents, f
        
    main(args, args.load_dset_pkl)
    