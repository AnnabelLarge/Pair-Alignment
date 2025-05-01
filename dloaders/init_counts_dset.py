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

from dloaders.CountsDset import CountsDset 
from dloaders.CountsDset import jax_collator as collator


def init_time_array(args):
    if args.pred_config['times_from'] == 'geometric':
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
        
    
    elif args.pred_config['times_from'] == 't_array_from_file':
        times_file = args.pred_config['filenames']['times']
        
        # read file
        times_from_array = []
        with open(f'{times_file}','r') as f:
            for line in f:
                times_from_array.append( float( line.strip() ) )
        times_from_array = np.array(times_from_array)
        single_time_from_file = False
    
    elif args.pred_config['times_from'] == 'one_time_per_sample_from_file':
        raise NotImplementedError('do you REALLY need an individual time per sample?')
    
    ### time cutoff
    # times_from_array = times_from_array[times_from_array > 1e-4] # error at beta approx
    # times_from_array = times_from_array[times_from_array > 1e-3] # error after jit compilation??? but fine without
    # final conclusion: use 0.015 as a cutoff
    times_from_array = times_from_array[times_from_array > args.min_time]
    
    return times_from_array, single_time_from_file 
        
   
def init_counts_dset(args, 
                     task, 
                     training_argparse=None):
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
        times_from_array, single_time_from_file = init_time_array(args)
        emission_alphabet_size = 4 if 'hky85' in args.pred_config['preset_name'] else 20
    
    
    #############################
    ### eval-specific options   #
    #############################
    elif task in ['eval']:
        only_test = True
        times_from_array, single_time_from_file = init_time_array(training_argparse)
        emission_alphabet_size = 4 if 'hky85' in training_argparse.pred_config['preset_name'] else 20
    
    
    #################
    ### LOAD DATA   #
    #################
    # test data
    print('Test dset:')
    for s in args.test_dset_splits:
        print(s)
    print()
    assert type(args.test_dset_splits) == list
    test_dset = CountsDset( data_dir = args.data_dir, 
                            split_prefixes = args.test_dset_splits,
                            single_time_from_file = single_time_from_file,
                            times_from_array = times_from_array,
                            emission_alphabet_size=emission_alphabet_size,
                            toss_alignments_longer_than = args.toss_alignments_longer_than,
                            bos_eos_as_match = args.bos_eos_as_match)

    # training data
    if not only_test:
        print('Training dset:')
        for s in args.train_dset_splits:
            print(s)
        print()
        
        assert type(args.train_dset_splits) == list
        training_dset = CountsDset( data_dir = args.data_dir, 
                                    split_prefixes = args.train_dset_splits,
                                    times_from_array = times_from_array,
                                    emission_alphabet_size=emission_alphabet_size,
                                    single_time_from_file = single_time_from_file,
                                    toss_alignments_longer_than = args.toss_alignments_longer_than,
                                    bos_eos_as_match = args.bos_eos_as_match)
    
    
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
