#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:36:42 2025

@author: annabel

About:
=======
initialize dataloaders and pytorch datasets

"""
import pickle
import torch
from torch.utils.data import DataLoader
import numpy as np
from argparse import Namespace

from dloaders.FullLenDset import FullLenDset
from dloaders.FullLenDset import jax_collator as collator
from dloaders.init_dataloader import init_dataloader
from dloaders.init_time_array import init_time_array


def init_full_len_dset( args: Namespace, 
                        task: str,
                        training_argparse: bool = None,
                        include_dataloader: bool = True ):
    """
    initialize the pytorch datasets and dataloaders (optional)
    """
    #################################
    ### training-specific options   #
    #################################
    if task in ['train', 
                'resume_train']:
        only_test = False
        t_per_sample = args.pred_config['times_from'] == 't_per_sample'
        t_array_for_all_samples = init_time_array(args)
        pred_model_type = args.pred_model_type
        gap_idx = args.gap_idx
        
        # if using a feedforward prediction head, enforce this value
        if pred_model_type == 'feedforward':
            args.times_from = None
        
        # if using markovian pairhmm, enforce this value
        elif pred_model_type in ['pairhmm_indp_sites',
                                 'pairhmm_frag_and_site_classes']:
            args.use_scan_fns = False
            
            if args.pred_config['subst_model_type'] == 'hky85':
                emission_alphabet_size = 4
            else:
                emission_alphabet_size = 20
        
        elif pred_model_type == 'neural_hmm':
            emission_alphabet_size = 20
        
    
    #############################
    ### eval-specific options   #
    #############################
    elif task in ['eval']:
        only_test = True
        t_per_sample = training_argparse.pred_config['times_from'] == 't_per_sample'
        t_array_for_all_samples = init_time_array(training_argparse)
        pred_model_type = training_argparse.pred_model_type
        gap_idx = training_argparse.gap_idx
        
        # if using a feedforward prediction head, enforce this value
        if pred_model_type == 'feedforward':
            args.times_from = None
        
        # if using markovian pairhmm, enforce this value
        elif pred_model_type in ['pairhmm_indp_sites',
                                 'pairhmm_frag_and_site_classes']:
            args.use_scan_fns = False
            
            if training_argparse.pred_config['subst_model_type'] == 'hky85':
                emission_alphabet_size = 4
            else:
                emission_alphabet_size = 20
        
        elif pred_model_type == 'neural_hmm':
            emission_alphabet_size = 20
            
    
    #################
    ### LOAD DATA   #
    #################
    # test data
    assert type(args.test_dset_splits) == list

    print('Test dset:')
    for s in args.test_dset_splits:
        print(s)
    print()

    test_dset = FullLenDset( data_dir = args.data_dir, 
                             split_prefixes = args.test_dset_splits,
                             pred_model_type = pred_model_type,
                             use_scan_fns = args.use_scan_fns,
                             t_per_sample = t_per_sample,
                             emission_alphabet_size=emission_alphabet_size,
                             chunk_length = args.chunk_length,
                             toss_alignments_longer_than = args.toss_alignments_longer_than,
                             seq_padding_idx = 0,
                             align_padding_idx = -9,
                             gap_idx = gap_idx
                             )
    out = {'test_dset': test_dset,
           't_array_for_all_samples': t_array_for_all_samples}
    
    # training data
    if not only_test:
        print('Training dset:')
        for s in args.train_dset_splits:
            print(s)
        print()
        
        assert type(args.train_dset_splits) == list
        training_dset = FullLenDset( data_dir = args.data_dir, 
                                     split_prefixes = args.train_dset_splits,
                                     pred_model_type = pred_model_type,
                                     use_scan_fns = args.use_scan_fns,
                                     t_per_sample = t_per_sample,
                                     emission_alphabet_size=emission_alphabet_size,
                                     chunk_length = args.chunk_length,
                                     toss_alignments_longer_than = args.toss_alignments_longer_than,
                                     seq_padding_idx = 0,
                                     align_padding_idx = -9,
                                     gap_idx = gap_idx
                                     )
        out['training_dset'] = training_dset
        
        
    ############################################
    ### create dataloaders, output dictionary  #
    ############################################
    if include_dataloader:
        test_dl = init_dataloader(args = args, 
                                  pytorch_custom_dset = test_dset,
                                  shuffle = False,
                                  collate_fn = collator)
        out['test_dl'] = test_dl
        
        if not only_test:
            training_dl = init_dataloader(args = args, 
                                          pytorch_custom_dset = training_dset,
                                          shuffle = True,
                                          collate_fn = collator)
            out['training_dl'] = training_dl
    
    return out
