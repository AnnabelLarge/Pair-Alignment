#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:33:01 2025

@author: annabel

train a pair hmm, under independent site class model assumption

"""
# general python
import os
import shutil
from tqdm import tqdm
from time import process_time
from time import time as wall_clock_time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # this is annoying
import pickle
from functools import partial
import platform
import argparse
import json
from contextlib import contextmanager

# jax/flax stuff
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

# pytorch imports
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# custom function/classes imports (in order of appearance)
from train_eval_fns.build_optimizer import build_optimizer
from utils.write_config import write_config
from utils.edit_argparse import enforce_valid_defaults
from utils.train_eval_utils import (setup_training_dir,
                                    timers,
                                    write_final_eval_results,
                                    record_postproc_time_table,
                                    pigz_compress_tensorboard_file)

# specific to training this model
from utils.edit_argparse import pairhmm_indp_sites_fill_with_default_values as fill_with_default_values
from utils.edit_argparse import pairhmms_share_top_level_args as share_top_level_args
from models.simple_site_class_predict.initializers import init_pairhmm_indp_sites as init_pairhmm
from train_eval_fns.TrainingWrapper import IndpSitesTrainingWrapper as TrainingWrapper
from train_eval_fns.indp_site_classes_training_fns import ( train_one_batch,
                                                            eval_one_batch,
                                                            final_eval_wrapper )


def train_pairhmm_indp_sites(args, dataloader_dict: dict):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    err = (f"{args.pred_model_type} is not pairhmm_indp_sites; "+
           f"using the wrong training script")
    assert args.pred_model_type == 'pairhmm_indp_sites', err
    del err
    
    # edit the argparse object in-place
    fill_with_default_values(args)
    enforce_valid_defaults(args)
    share_top_level_args(args)
    
    if not args.update_grads:
        print('DEBUG MODE: DISABLING GRAD UPDATES')

    
    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    ### initial setup of misc things
    # setup the working directory (if not done yet) and this run's sub-directory
    setup_training_dir(args)
    
    # initial random key, to carry through execution
    rngkey = jax.random.key(args.rng_seednum)
    
    # setup tensorboard writer
    writer = SummaryWriter(args.tboard_dir)
    
    # create a new logfile with header
    with open(args.logfile_name,'w') as g:
        # disabled training
        if not args.update_grads:
            g.write('DEBUG MODE: DISABLING GRAD UPDATES\n\n')
        
        # standard header
        g.write( f'PairHMM with independent site classes over emissions\n' )
        g.write( f'Substitution model: {args.pred_config["subst_model_type"]}\n' )
        g.write( f'Indel model: {args.pred_config.get("indel_model_type","None")}\n' )
        g.write( (f'  - Number of site classes for emissions: '+
                  f'{args.pred_config["num_site_mixtures"]}\n' +
                  f'  - Possible substitution rate multipliers: ' +
                  f'{args.pred_config["k_rate_mults"]}\n')
                )
        
        # note if rates are independent
        if args.pred_config['indp_rate_mults']:
            possible_rates =  args.pred_config['k_rate_mults']
            g.write( (f'  - Rates are INDEPENDENT FROM site class label: '+
                      f'( P(k | c) = P(k) ); {possible_rates} possible '+
                      f'rate multipliers\n' )
                    )
                    
        elif not args.pred_config['indp_rate_mults']:
            possible_rates = (args.pred_config['num_domain_mixtures'] *
                              args.pred_config['num_fragment_mixtures'] *
                              args.pred_config['num_site_mixtures'] *
                              args.pred_config['k_rate_mults'] )
            g.write( ( f'  - Rates depend on other mixtures ( P(k | c) ); '+
                       f'{possible_rates} possible rate multipliers\n' )
                    )
        
        # how to normalize reported metrics (usually by descendant length)
        if args.pred_config["indel_model_type"] is not None:
            g.write( f'  - When reporting, normalizing losses by: {args.norm_reported_loss_by}\n' )
        
        elif args.pred_config["indel_model_type"] is None:
            g.write( f'  - When reporting, normalizing losses by: align length '+
                     f'(same as desc length, because we remove gap '+
                     f'positions) \n' )
        
        # write source of times
        g.write( f'Times from: {args.pred_config["times_from"]}\n' )
    
    
    # extra files to record if you use tkf approximations
    with open(f'{args.out_arrs_dir}/TRAIN_tkf_approx.tsv','w') as g:
        g.write('Used tkf approximations in the following locations:\n')
    
    with open(f'{args.out_arrs_dir}/FINAL-EVAL_tkf_approx.tsv','w') as g:
        g.write('Used tkf approximations in the following locations:\n')
        
        
    ### save updated config, provide filename for saving model parameters
    finalpred_save_model_filename = args.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
    write_config(args = args, out_dir = args.model_ckpts_dir)
    
    
    ### extract data from dataloader_dict
    training_dset = dataloader_dict['training_dset']
    training_dl = dataloader_dict['training_dl']
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    t_array_for_all_samples = dataloader_dict['t_array_for_all_samples']
    args.pred_config['training_dset_emit_counts'] = training_dset.emit_counts
    
    
    
    ###########################################################################
    ### 2: MODEL INIT, TRAINING  ##############################################
    ###########################################################################
    print('2: model init')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'2: model init\n')
    
    
    # init the optimizer, split a new rng key
    tx = build_optimizer(args)
    rngkey, model_init_rngkey = jax.random.split(rngkey, num=2)
    
    
    ### determine shapes for init
    B = args.batch_size
    A = args.emission_alphabet_size
    S = training_dset.num_transitions
    
    # time
    if t_array_for_all_samples is not None:
        dummy_t_array_for_all_samples = jnp.empty( (t_array_for_all_samples.shape[0], ) )
        dummy_t_for_each_sample = None
    
    else:
        dummy_t_array_for_all_samples = None
        dummy_t_for_each_sample = jnp.empty( (B,) )
        
    # counts array
    dummy_subCounts = jnp.empty( (B, A, A) )
    dummy_insCounts = jnp.empty( (B, A) )
    dummy_delCounts = jnp.empty( (B, A) )
    dummy_transCounts = jnp.empty( (B, S, S) )
    
    fake_batch = [dummy_subCounts,
                  dummy_insCounts,
                  dummy_delCounts,
                  dummy_transCounts,
                  dummy_t_for_each_sample]
    
    
    ### initialize functions
    out = init_pairhmm( seq_shapes = fake_batch, 
                        dummy_t_array = dummy_t_array_for_all_samples,
                        tx = tx, 
                        model_init_rngkey = model_init_rngkey,
                        pred_config = args.pred_config,
                        tabulate_file_loc = args.model_ckpts_dir
                        )
    pairhmm_trainstate, pairhmm_instance = out
    del out
    
    ### part+jit training function
    if t_array_for_all_samples is not None:
        print('Using times:')
        print(t_array_for_all_samples)
        print()
    
    elif t_array_for_all_samples is None:
        print('Using one branch length per sample')
        print()
    
    parted_train_fn = partial( train_one_batch,
                               indel_model_type = args.pred_config.get('indel_model_type', None),
                               t_array = t_array_for_all_samples, 
                               interms_for_tboard = args.interms_for_tboard,
                               whole_dset_grad_desc = False,
                               update_grads = args.update_grads )
    train_fn_jitted = jax.jit(parted_train_fn)
    del parted_train_fn
    
    
    ### part+jit eval function
    no_outputs = {k: False for k in args.interms_for_tboard.keys()}
    parted_eval_fn = partial( eval_one_batch,
                              t_array = t_array_for_all_samples,
                              pairhmm_instance = pairhmm_instance,
                              interms_for_tboard = no_outputs,
                              return_all_loglikes = False )
    eval_fn_jitted = jax.jit(parted_eval_fn)
    del parted_eval_fn
    
    
    ### initialize training wrapper
    training_wrapper = TrainingWrapper( args = args,
                                        epoch_arr = range(args.num_epochs),
                                        initial_training_rngkey = rngkey,
                                        dataloader_dict = dataloader_dict,
                                        train_fn_jitted = train_fn_jitted,
                                        eval_fn_jitted = eval_fn_jitted,
                                        all_save_model_filenames = [finalpred_save_model_filename],
                                        writer = writer )
    
    
    ### train
    print(f'3: main training loop')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'3: main training loop\n')
    
    out = training_wrapper.run_train_loop( all_trainstates = [pairhmm_trainstate] )
    early_stop, best_epoch, best_trainstates = out
    del out
    
    best_pairhmm_trainstate = best_trainstates[0]
    del best_trainstates
    
    
    
    ###########################################################################
    ### FINAL EVAL   ##########################################################
    ###########################################################################
    print(f'4: post-training actions')
    # write to logfile
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'4: post-training actions\n')
    
    # don't accidentally use old trainstates or eval fn
    del pairhmm_trainstate, eval_fn_jitted
    
    # new timer for these steps
    postproc_timer_class = timers( num_epochs = 1 )
    postproc_timer_class.start_timer()


    ### write to output logfile
    with open(args.logfile_name,'a') as g:
        # if early stopping was never triggered, record results at last epoch
        if not early_stop:
            g.write(f'Regular stopping after {args.num_epochs} full epochs:\n\n')
        
        # finish up logfile, regardless of early stopping or not
        g.write(f'Epoch with lowest average test loss ("best epoch"): {best_epoch}\n')
        g.write(f'RE-EVALUATING ALL DATA WITH BEST PARAMS\n\n')
    

    ### save the argparse object by itself
    args.epoch_idx = best_epoch
    with open(f'{args.model_ckpts_dir}/TRAINING_ARGPARSE.pkl', 'wb') as g:
        pickle.dump(args, g)
        
    del best_epoch, early_stop


    ### jit compile new eval function
    parted_eval_fn = partial( eval_one_batch,
                              t_array = t_array_for_all_samples,
                              all_trainstates = [best_pairhmm_trainstate],
                              pairhmm_instance = pairhmm_instance,
                              interms_for_tboard = args.interms_for_tboard,
                              return_all_loglikes = True )
    eval_fn_jitted = jax.jit(parted_eval_fn)
    del parted_eval_fn
    

    ###########################################
    ### loop through training dataloader and  #
    ### score with best params                #
    ###########################################
    with open(args.logfile_name,'a') as g:
        g.write(f'SCORING ALL TRAIN SEQS\n\n')
        
    train_summary_stats = final_eval_wrapper(dataloader = training_dl, 
                                             dataset = training_dset, 
                                             eval_fn_jitted = eval_fn_jitted,
                                             save_per_sample_losses = args.save_per_sample_losses,
                                             logfile_dir = args.logfile_dir,
                                             out_arrs_dir = args.out_arrs_dir,
                                             outfile_prefix = f'train-set')


    ###########################################
    ### loop through test dataloader and      #
    ### score with best params                #
    ###########################################
    with open(args.logfile_name,'a') as g:
        g.write(f'SCORING ALL TEST SEQS\n\n')
        
    test_summary_stats = final_eval_wrapper(dataloader = test_dl, 
                                            dataset = test_dset,  
                                            eval_fn_jitted = eval_fn_jitted,
                                            save_per_sample_losses = args.save_per_sample_losses,
                                            logfile_dir = args.logfile_dir,
                                            out_arrs_dir = args.out_arrs_dir,
                                            outfile_prefix = f'test-set')
    
    
    ### un-transform parameters and write to numpy arrays
    # if using one branch length per sample, write arrays with the test set
    if t_array_for_all_samples is not None:
        best_pairhmm_trainstate.apply_fn( variables = best_pairhmm_trainstate.params,
                                          t_array = t_array_for_all_samples,
                                          prefix = '',
                                          out_folder = args.out_arrs_dir,
                                          write_time_static_objs = True,
                                          method = pairhmm_instance.write_params )
        
    elif t_array_for_all_samples is None:
        t_arr = test_dset.times
        
        pt_id = 0
        for i in tqdm( range(0, t_arr.shape[0], args.batch_size) ):
            batch_t = jnp.array( t_arr[i : (i + args.batch_size)] )
            batch_prefix = f'test-set_pt{pt_id}'
            best_pairhmm_trainstate.apply_fn( variables = best_pairhmm_trainstate.params,
                                              t_array = batch_t,
                                              prefix = batch_prefix,
                                              out_folder = args.out_arrs_dir,
                                              write_time_static_objs = (pt_id==0),
                                              method = pairhmm_instance.write_params )
            
            pt_id += 1
        
    

    ###########################################
    ### update the logfile with final losses  #
    ###########################################
    write_final_eval_results(args = args, 
                             summary_stats = train_summary_stats,
                             filename = 'TRAIN-AVE-LOSSES.tsv')

    write_final_eval_results(args = args, 
                             summary_stats = test_summary_stats,
                             filename = 'TEST-AVE-LOSSES.tsv')

    # record total time spent on post-training actions; write this to a table
    #   instead of a scalar collection
    record_postproc_time_table( already_started_timer_class = postproc_timer_class,
                                writer = writer )

    # when you're done with the function, close the tensorboard writer and
    #   compress the output file
    writer.close()
    pigz_compress_tensorboard_file( args )
    
