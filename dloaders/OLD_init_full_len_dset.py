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


def init_full_len_dset_hmm( args: Namespace, 
                            task: str,
                            training_argparse = None,
                            include_dataloader: bool = True ):
    """
    initialize the pytorch datasets and dataloaders for pairHMM models
    
    use train/test split
    """
    
    ### behavior that depends on task
    if task in ['train', 'resume_train']:
        argparse_obj = args
        only_test = False
    
    elif task == 'eval':
        argparse_obj = training_argparse
        only_test = True
    
    pred_model_type = argparse_obj.pred_model_type
    gap_idx = argparse_obj.gap_idx
    
    ### enforce defaults about emission alphabet size
    if argparse_obj.pred_config['subst_model_type'] == 'hky85':
        emission_alphabet_size = 4
    else:
        emission_alphabet_size = 20
    
    # regular pairhmm doesn't use scan functions
    argparse_obj.use_scan_fns = False
    
    
    ### handle times: either a grid of times for all samples (T,) or a unique
    ###   branch length for every sample (B,)
    cond1 = argparse_obj.pred_config['times_from'] is None
    cond2 = ( argparse_obj.pred_config['times_from'] == 't_per_sample' )
    t_per_sample = cond1 or cond2
    del cond1, cond2
    
    # init a grid if t_per_sample is False
    if t_per_sample:
        t_array_for_all_samples = None
        
    elif not t_per_sample:
        t_array_for_all_samples = init_time_array( argparse_obj )
    
    # no longer need this
    del argparse_obj
    
    
    #################
    ### LOAD DATA   #
    #################
    # test data
    assert isinstance(args.test_dset_splits, list)

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
        
        assert isinstance(args.train_dset_splits, list)
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


def init_full_len_dset_neural( args: Namespace, 
                               task: str,
                               training_argparse = None,
                               include_dataloader: bool = True ):
    """
    initialize the pytorch datasets and dataloaders
    
    use train/dev/test split, because neural models need dev set
    """
    
    ### behavior that depends on task
    if task in ['train', 'resume_train']:
        argparse_obj = args
        only_test = False
    
    elif task == 'eval':
        argparse_obj = training_argparse
        only_test = True
    
    pred_model_type = argparse_obj.pred_model_type
    gap_idx = argparse_obj.gap_idx
    
    
    ### enforce defaults: feedforward to alignment-augmented alphabet
    if pred_model_type == 'feedforward':
        # only protein model implemented for now
        emission_alphabet_size = 20
        
        # remap values
        if argparse_obj.pred_config['t_per_sample']:
            argparse_obj.pred_config['times_from'] = 't_per_sample'
        
        elif not argparse_obj.pred_config['t_per_sample']:
            argparse_obj.pred_config['times_from'] = None
            
            
    ### enforce defaults: markovian alignment algorithms
    elif pred_model_type == 'neural_hmm':
        # enforce defaults about emission alphabet size
        if argparse_obj.pred_config['subst_model_type'] == 'hky85':
            emission_alphabet_size = 4
        else:
            emission_alphabet_size = 20
    
    
    ### handle times: either a grid of times for all samples (T,) or a unique
    ###   branch length for every sample (B,)
    cond1 = argparse_obj.pred_config['times_from'] is None
    cond2 = ( argparse_obj.pred_config['times_from'] == 't_per_sample' )
    t_per_sample = cond1 or cond2
    del cond1, cond2
    
    # init a grid if t_per_sample is False
    if t_per_sample:
        t_array_for_all_samples = None
        
    elif not t_per_sample:
        t_array_for_all_samples = init_time_array( argparse_obj )
    
    # no longer need this
    del argparse_obj
    
    
    #################
    ### LOAD DATA   #
    #################
    # test data
    assert isinstance(args.test_dset_splits, list)

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
    
    if not only_test:
        # dev set data: for hyperparam tuning
        print('Dev dset:')
        for s in args.dev_dset_splits:
            print(s)
        print()
        
        assert isinstance(args.dev_dset_splits, list)
        dev_dset = FullLenDset( data_dir = args.data_dir, 
                                split_prefixes = args.dev_dset_splits,
                                pred_model_type = pred_model_type,
                                use_scan_fns = args.use_scan_fns,
                                t_per_sample = t_per_sample,
                                emission_alphabet_size=emission_alphabet_size,
                                chunk_length = args.chunk_length,
                                toss_alignments_longer_than = args.toss_alignments_longer_than,
                                seq_padding_idx = 0,
                                align_padding_idx = -9,
                                gap_idx = gap_idx )
        out['dev_dset'] = dev_dset
        
        # training data
        print('Training dset:')
        for s in args.train_dset_splits:
            print(s)
        print()
        
        assert isinstance(args.train_dset_splits, list)
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
            dev_dl = init_dataloader(args = args, 
                                     pytorch_custom_dset = dev_dset,
                                     shuffle = True,
                                     collate_fn = collator)
            out['dev_dl'] = dev_dl
            
            training_dl = init_dataloader(args = args, 
                                          pytorch_custom_dset = training_dset,
                                          shuffle = True,
                                          collate_fn = collator)
            out['training_dl'] = training_dl
    
    return out
