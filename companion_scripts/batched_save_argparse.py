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

from save_argparse import main as save_argparse_fn
from save_argparse import load_dset_pkl_fn

def read_config_file(config_file):
    with open(config_file, 'r') as f:
        contents = json.load(f)
        t_args = argparse.Namespace()
        t_args.__dict__.update(contents)
        args = parser.parse_args(namespace=t_args)
    return args

def main(args):
    # read argparse from first config file
    file_lst = [file for file in os.listdir(args.configs) if not file.startswith('.')
                and file.endswith('.json')]
    assert len(file_lst) > 0, f'{args.configs} is empty!'
    
    # get dataloader and functions from first config file
    first_config_file = file_lst[0]
    print(f'DATALOADER CONSTRUCTED FROM: {args.configs}/{first_config_file}')
    print(f"WARNING: make sure you want this dataloader for ALL experiments in {args.configs}!!!")
    first_args = read_config_file(f'{args.configs}/{first_config_file}')
    pred_model_type = first_args.pred_model_type
    
    # import correct wrappers, dataloader initializers
    if pred_model_type == 'pairhmm_indp_sites':
        from dloaders.init_counts_dset import init_counts_dset as init_datasets
        from dloaders.CountsDset import jax_collator as collate_fn
        
    elif pred_model_type in ['pairhmm_frag_and_site_classes',
                                  'neural_hmm',
                                  'feedforward']:
        from dloaders.init_full_len_dset import init_full_len_dset as init_datasets
        from dloaders.FullLenDset import jax_collator as collate_fn

    # load data
    if args.load_dset_pkl is None:
        dload_dict_for_all = init_datasets( first_args,
                                            'train',
                                            training_argparse = None,
                                            include_dataloader = True )
    else:
        dload_dict_for_all = load_dset_pkl_fn(args = first_args,
                                              file_to_load = args.load_dset_pkl,
                                              collate_fn = collate_fn)
        
    # with this dload_dict, convert all
    for file in file_lst:
        # read argparse
        assert file.endswith('.json'), "input is one JSON file"
        this_run_args = read_config_file(f'{args.configs}/{file}')
        print(f'CONVERTING: {args.configs}/{file}')
    
        this_run_args.pred_config['training_dset_emit_counts'] = dload_dict_for_all['training_dset'].emit_counts
        fill_with_default_values(this_run_args)
        enforce_valid_defaults(this_run_args)
        share_top_level_args(this_run_args)
        
        # missing best epoch, but oh well
        model_ckpts_dir = f'{os.getcwd()}/{this_run_args.training_wkdir}/model_ckpts'
        with open(f'{model_ckpts_dir}/TRAINING_ARGPARSE.pkl', 'wb') as g:
            pickle.dump(this_run_args, g)
        

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
    main(args)
    