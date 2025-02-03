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
import random
from torch.utils.data import DataLoader
import numpy as np


def init_time_array(args):
    # feedforward prediction head with no times
    if args.pred_model_type == 'feedforward':
        times_from_array = None
        single_time_from_file = False
    
    # pairHMM models, which use times
    elif (args.pred_model_type != 'feedforward') and (args.times_from == 'geometric'):
        t_grid_center = args.pred_config['t_grid_center']
        t_grid_step = args.pred_config['t_grid_step']
        t_grid_num_steps = args.pred_config['t_grid_num_steps']
        
        quantization_grid = range( -(t_grid_num_steps-1), 
                                   t_grid_num_steps, 
                                   1
                                  )
        times_from_array = np.array([ (t_grid_center * t_grid_step**q_i) 
                                      for q_i in quantization_grid
                                     ]
                                    )
        single_time_from_file = False
    
    elif (args.pred_model_type != 'feedforward') and (args.times_from == 't_array_from_file'):
        times_file = args.pred_config['times_file']
        const_for_time_marg = args.pred_config['const_for_time_marg']
        
        # read file
        times_from_array = []
        with open(f'{times_file}','r') as f:
            for line in f:
                times_from_array.append( float( line.strip() ) )
        times_from_array = np.array(times_from_array)
        
        # use t_grid_step argument for time marginalization
        #   modify args in place to do renaming
        args.pred_config['t_grid_step'] = const_for_time_marg
        
        single_time_from_file = False
        
    elif (args.pred_model_type != 'feedforward') and (args.times_from == 'one_time_per_sample_from_file'):
        raise NotImplementedError('do you REALLY need an individual time per sample?')
    
    return (times_from_array, single_time_from_file)
        
   
def init_dataloaders(args, task):
    """
    initialize the dataloaders
    """
    #########################################################
    ### set random seeds for numpy and pytorch separately   #
    #########################################################
    torch.manual_seed(args.rng_seednum)
    random.seed(args.rng_seednum)
    np.random.seed(args.rng_seednum)
    
    
    #################################
    ### training-specific options   #
    #################################
    if task in ['train', 
                'resume_train']:
        # if using a feedforward prediction head, enforce this value
        if args.pred_model_type == 'feedforward':
            args.times_from = None
        
        # misc params, time array
        times_from_array, single_time_from_file = init_time_array(args)
        only_test = False
        pred_model_type = args.pred_model_type
    
    
    #############################
    ### eval-specific options   #
    #############################
    elif task in ['eval']:
        ### load the training argparse
        training_argparse_filename = (f'{args.training_wkdir}/'+
                                      f'model_ckpts/TRAINING_ARGPARSE.pkl')
        
        with open(training_argparse_filename,'rb') as g:
            training_argparse = pickle.load(g)
        
        
        ### use values from training argparse to set values
        times_from_array, single_time_from_file = init_time_array(training_argparse)
        only_test = True
        pred_model_type = training_argparse.pred_model_type
    
    
    #######################################
    ### OPTION 1: use the full sequence   #
    #######################################
    if args.pred_model_type in ['feedforward', 'neural_pairhmm']:
        from dloaders.FullLenDset import FullLenDset
        from dloaders.FullLenDset import jax_collator as collator
        
        # test data
        print(f'Test dset: {args.test_dset_splits}')
        assert type(args.test_dset_splits) == list
        test_dset = FullLenDset( data_dir = args.data_dir, 
                                 split_prefixes = args.test_dset_splits,
                                 pred_model_type = pred_model_type,
                                 use_scan_fns = args.use_scan_fns,
                                 times_from_array = times_from_array,
                                 single_time_from_file = single_time_from_file,
                                 chunk_length = args.chunk_length,
                                 toss_alignments_longer_than = args.toss_alignments_longer_than,
                                 seq_padding_idx = 0,
                                 align_padding_idx = -9,
                                 gap_tok = 43,
                                 emission_alphabet_size = 20
                                 )
        
        # training data
        if not only_test:
            print(f'Training dset: {args.train_dset_splits}')
            assert type(args.train_dset_splits) == list
            training_dset = FullLenDset( data_dir = args.data_dir, 
                                         split_prefixes = args.train_dset_splits,
                                         pred_model_type = args.pred_model_type,
                                         use_scan_fns = args.use_scan_fns,
                                         times_from_array = times_from_array,
                                         single_time_from_file = single_time_from_file,
                                         chunk_length = args.chunk_length,
                                         toss_alignments_longer_than = args.toss_alignments_longer_than,
                                         seq_padding_idx = 0,
                                         align_padding_idx = -9,
                                         gap_tok = 43,
                                         emission_alphabet_size = 20
                                         )
    
    
    ######################################################
    ### OPTION 2: use summaries of counts, precomputed   #
    ######################################################
    elif args.pred_model_type in ['pairhmm']:
        from dloaders.CountsDset import CountsDset 
        from dloaders.CountsDset import jax_collator as collator
        
        # test data
        print(f'Test dset: {args.test_dset_splits}')
        assert type(args.test_dset_splits) == list
        test_dset = CountsDset( data_dir = args.data_dir, 
                                split_prefixes = args.test_dset_splits,
                                subsOnly = args.subsOnly,
                                single_time_from_file = single_time_from_file,
                                times_from_array = times_from_array,
                                toss_alignments_longer_than = args.toss_alignments_longer_than
                                )
    
        # training data
        if not only_test:
            print(f'Training dset: {args.train_dset_splits}')
            assert type(args.train_dset_splits) == list
            training_dset = CountsDset( data_dir = args.data_dir, 
                                        split_prefixes = args.train_dset_splits,
                                        subsOnly = args.subsOnly,
                                        times_from_array = times_from_array,
                                        single_time_from_file = single_time_from_file,
                                        toss_alignments_longer_than = args.toss_alignments_longer_than
                                        )
    
    ############################################
    ### create dataloaders, output dictionary  #
    ############################################
    test_dl = DataLoader( test_dset, 
                          batch_size = args.batch_size, 
                          shuffle = False,
                          collate_fn = collator
                         )
    
    out = {'test_dset': test_dset,
           'test_dl': test_dl}
    
    if not only_test:
        training_dl = DataLoader( training_dset, 
                                  batch_size = args.batch_size, 
                                  shuffle = True,
                                  collate_fn = collator
                                 )
        
        out['training_dset'] = training_dset
        out['training_dl'] = training_dl
    
    return out
