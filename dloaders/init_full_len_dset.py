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

from dloaders.FullLenDset import FullLenDset
from dloaders.FullLenDset import jax_collator as collator
from dloaders.init_dataloader import init_dataloader


def init_time_array(args):
    ### when there's no times to return
    if (args.pred_model_type == 'feedforward') or (args.pred_config['times_from'] == 't_per_sample'):
        return None
    
    
    ### init from geometric grid, like in cherryML
    elif (args.pred_model_type != 'feedforward') and args.pred_config['times_from'] == 'geometric':
        t_grid_center = args.pred_config['t_grid_center']
        t_grid_step = args.pred_config['t_grid_step']
        t_grid_num_steps = args.pred_config['t_grid_num_steps']
        
        quantization_grid = range( -(t_grid_num_steps-1), 
                                   t_grid_num_steps, 
                                   1
                                  )
        t_array = [ (t_grid_center * t_grid_step**q_i) for q_i in quantization_grid ]
        
        return np.array(t_array)
    
    
    ### read times from flat text file
    elif (args.pred_model_type != 'feedforward') and args.pred_config['times_from'] == 't_array_from_file':
        times_file = args.pred_config['filenames']['times']
        
        # read file
        t_array = []
        with open(f'{times_file}','r') as f:
            for line in f:
                t_array.append( float( line.strip() ) )
        
        return np.array(t_array)
    
    
    ### figure out time quantization per sample... later
    elif (args.pred_model_type != 'feedforward') and args.pred_config['times_from'] == 't_quantized_per_sample':
        raise NotImplementedError    
   
    
def init_full_len_dset( args, 
                        task,
                        training_argparse=None,
                        include_dataloader=True ):
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
            
            if args.pred_config['subst_model_type'].lower() == 'hky85':
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
            training_argparse.times_from = None
        
        # if using markovian pairhmm, enforce this value
        elif pred_model_type in ['pairhmm_indp_sites',
                                 'pairhmm_frag_and_site_classes']:
            training_argparse.use_scan_fns = False
            
            if training_argparse.pred_config['subst_model_type'].lower() == 'hky85':
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
