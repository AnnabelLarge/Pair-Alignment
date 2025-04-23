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

from dloaders.FullLenDset import FullLenDset
from dloaders.FullLenDset import jax_collator as collator


def init_time_array(args):
    # feedforward prediction head with no times
    if args.pred_model_type == 'feedforward':
        times_from_array = None
        single_time_from_file = False
    
    # pairHMM models, which use times
    elif (args.pred_model_type != 'feedforward') and (args.pred_config['times_from'] == 'geometric'):
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
    
    elif (args.pred_model_type != 'feedforward') and (args.pred_config['times_from'] == 't_array_from_file'):
        try:
            times_file = args.pred_config['filenames']['times']
        except:
            times_file = args.pred_config['times_file']
        
        # read file
        times_from_array = []
        with open(f'{times_file}','r') as f:
            for line in f:
                times_from_array.append( float( line.strip() ) )
        times_from_array = np.array(times_from_array)
        
        single_time_from_file = False
        
    elif (args.pred_model_type != 'feedforward') and (args.pred_config['times_from'] == 'one_time_per_sample_from_file'):
        raise NotImplementedError('do you REALLY need an individual time per sample?')
    
    return (times_from_array, single_time_from_file)
        
   
    
def init_full_len_dset( args, 
                        task,
                        training_argparse=None ):
    """
    initialize the dataloaders
    """
    

    #################################
    ### training-specific options   #
    #################################
    if task in ['train', 
                'resume_train']:
        torch.manual_seed(args.rng_seednum)
        random.seed(args.rng_seednum)
        np.random.seed(args.rng_seednum)    
        only_test = False
        
        pred_model_type = args.pred_model_type
        
        # if using a feedforward prediction head, enforce this value
        if pred_model_type == 'feedforward':
            args.times_from = None
        
        # if using markovian pairhmm, enforce this value
        if pred_model_type.startswith('pairhmm'):
            args.use_scan_fns = False
        
        # misc params, time array
        times_from_array, single_time_from_file = init_time_array(args)

        # emission alphabet size
        emission_alphabet_size = 4 if 'hky85' in args.pred_config['preset_name'] else 20
    
    
    #############################
    ### eval-specific options   #
    #############################
    elif task in ['eval']:
        only_test = True

        ### use values from training argparse to set values
        pred_model_type = training_argparse.pred_model_type
        
        # if using a feedforward prediction head, enforce this value
        if pred_model_type == 'feedforward':
            args.times_from = None
        
        # if using markovian pairhmm, enforce this value
        if pred_model_type.startswith('pairhmm'):
            args.use_scan_fns = False
        
        times_from_array, single_time_from_file = init_time_array(training_argparse)

        # emission alphabet size
        emission_alphabet_size = 4 if 'hky85' in args.pred_config['preset_name'] else 20
        
    
    
    #################
    ### LOAD DATA   #
    #################
    # test data
    print('Test dset:')
    for s in args.test_dset_splits:
        print(s)
    print()
    assert type(args.test_dset_splits) == list
    test_dset = FullLenDset( data_dir = args.data_dir, 
                             split_prefixes = args.test_dset_splits,
                             pred_model_type = pred_model_type,
                             use_scan_fns = args.use_scan_fns,
                             times_from_array = times_from_array,
                             emission_alphabet_size=emission_alphabet_size,
                             single_time_from_file = single_time_from_file,
                             chunk_length = args.chunk_length,
                             toss_alignments_longer_than = args.toss_alignments_longer_than,
                             seq_padding_idx = 0,
                             align_padding_idx = -9,
                             gap_idx = 43
                             )
    
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
                                     times_from_array = times_from_array,
                                     emission_alphabet_size=emission_alphabet_size,
                                     single_time_from_file = single_time_from_file,
                                     chunk_length = args.chunk_length,
                                     toss_alignments_longer_than = args.toss_alignments_longer_than,
                                     seq_padding_idx = 0,
                                     align_padding_idx = -9,
                                     gap_idx = 43
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
