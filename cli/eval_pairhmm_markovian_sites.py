#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:55:46 2025

Load parameters and evaluate likelihoods for an markovian
  site class model

"""
# general python
import os
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import pickle
from functools import partial
import platform
import argparse
import json

# jax/flax stuff
import jax
import jax.numpy as jnp
import flax

# pytorch imports
from torch.utils.data import DataLoader

# custom function/classes imports (in order of appearance)
from train_eval_fns.build_optimizer import build_optimizer
from utils.sequence_length_helpers import determine_alignlen_bin
from models.simple_site_class_predict.initializers import init_pairhmm_markov_sites as init_pairhmm
from train_eval_fns.markovian_site_classes_training_fns import ( eval_one_batch,
                                                                 final_eval_wrapper )

from utils.edit_argparse import (enforce_valid_defaults,
                                 fill_with_default_values,
                                 share_top_level_args)


def eval_pairhmm_markovian_sites( args, 
                                  dataloader_dict: dict,
                                  training_argparse ):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    # final where model pickles, previous argparses are
    err = (f"{training_argparse.pred_model_type} is not pairhmm_markovian_sites; "+
           f"using the wrong eval script")
    assert training_argparse.pred_model_type == 'pairhmm_markovian_sites', err
    del err
        
    prev_model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
    pairhmm_savemodel_filename = prev_model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
    
    fill_with_default_values(training_argparse)
    enforce_valid_defaults(training_argparse)
    share_top_level_args(training_argparse)

    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    ### create the eval working directory, if it doesn't exist
    args.logfile_dir = f'{args.eval_wkdir}/logfiles'
    args.logfile_name = f'{args.logfile_dir}/PROGRESS.log'
    args.out_arrs_dir = f'{args.eval_wkdir}/out_arrs'    
    args.model_ckpts_dir = f'{args.eval_wkdir}/model_ckpts'
    if args.eval_wkdir not in os.listdir():
        os.mkdir(args.eval_wkdir)
        os.mkdir(args.logfile_dir)
        os.mkdir(args.out_arrs_dir)
        os.mkdir(args.model_ckpts_dir)
    
    # create a new logfile
    with open(args.logfile_name,'w') as g:
        g.write( f'Loading from {args.training_wkdir} to eval new data\n' )
        g.write( (f'  - Number of site classes for emissions and transitions: '+
                  f'{training_argparse.pred_config["num_emit_site_classes"]}\n' )
                )
        g.write( f'  - Normalizing losses by: {training_argparse.norm_loss_by}\n' )
    
    
    ### extract data from dataloader_dict
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    
    
    ###########################################################################
    ### 1: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
    ###########################################################################
    print('1: model init')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'1: model init\n')
    
    
    # need to intialize an optimizer for compatibility when restoring the state, 
    #   but we're not training so this doesn't really matter?
    tx = build_optimizer(training_argparse)
    
    
    ### determine shapes for init
    # time
    num_timepoints = test_dset.retrieve_num_timepoints(times_from = training_argparse.pred_config['times_from'])
    dummy_t_array = jnp.empty( (num_timepoints, ) )
    
    
    ### init sizes
    # (B, L, 3)
    max_dim1 = test_dset.global_align_max_length 
    largest_aligns = jnp.empty( (args.batch_size, max_dim1, 3), dtype=int )
    del max_dim1
    
    ### fn to handle jit-compiling according to alignment length
    parted_determine_alignlen_bin = partial(determine_alignlen_bin,  
                                            chunk_length = args.chunk_length,
                                            seq_padding_idx = training_argparse.seq_padding_idx)
    jitted_determine_alignlen_bin = jax.jit(parted_determine_alignlen_bin)
    del parted_determine_alignlen_bin
    
    
    ### initialize functions
    out = init_pairhmm( seq_shapes = largest_aligns, 
                        dummy_t_array = dummy_t_array,
                        tx = tx, 
                        model_init_rngkey = jax.random.key(0),
                        pred_config = training_argparse.pred_config,
                        tabulate_file_loc = args.model_ckpts_dir)
    blank_tstate, pairhmm_instance = out
    del out
    
    # load values
    with open(pairhmm_savemodel_filename, 'rb') as f:
        state_dict = pickle.load(f)
        
    best_pairhmm_trainstate = flax.serialization.from_state_dict( blank_tstate, 
                                                                  state_dict )
    del blank_tstate, state_dict
    
    
    ### part+jit eval function
    t_array = test_dset.return_time_array()
    null_interms_dict = {k: False for k in training_argparse.interms_for_tboard.keys()}
    parted_eval_fn = partial( eval_one_batch,
                              t_array = t_array,
                              pairhmm_trainstate = best_pairhmm_trainstate,
                              pairhmm_instance = pairhmm_instance,
                              interms_for_tboard = null_interms_dict,
                              return_all_loglikes = True )
    
    eval_fn_jitted = jax.jit(parted_eval_fn, 
                              static_argnames = ['max_align_len'])
    del parted_eval_fn
    
    
    ### write the parameters again
    best_pairhmm_trainstate.apply_fn( variables = best_pairhmm_trainstate.params,
                                      t_array = t_array,
                                      out_folder = args.out_arrs_dir,
                                      method = pairhmm_instance.write_params )
    
    
    ###########################################################################
    ### 2: EVAL   #############################################################
    ###########################################################################
    print(f'2: eval')
    # write to logfile
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'2: eval\n')

    test_summary_stats = final_eval_wrapper(dataloader = test_dl, 
                                            dataset = test_dset, 
                                            eval_fn_jitted = eval_fn_jitted,
                                            save_per_sample_losses = True,
                                            jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                                            logfile_dir = args.logfile_dir,
                                            out_arrs_dir = args.out_arrs_dir,
                                            outfile_prefix = f'dset')
    
    
    ###########################################
    ### update the logfile with final losses  #
    ###########################################
    # save the trainstate again
    with open(f'{args.model_ckpts_dir}/FINAL_PRED.pkl', 'wb') as g:
        model_state_dict = flax.serialization.to_state_dict(best_pairhmm_trainstate)
        pickle.dump(model_state_dict, g)
        
    to_write = {'RUN': args.training_wkdir}
    to_write = {**to_write, **test_summary_stats}
    
    with open(f'{args.logfile_dir}/AVE-LOSSES.tsv','w') as g:
        for k, v in to_write.items():
            g.write(f'{k}\t{v}\n')
    