#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:45:05 2025

@author: annabel
"""
# general python
import os
import shutil
import glob
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

# jax/flax stuff
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

# pytorch imports
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# custom function/classes imports
from train_eval_fns.build_optimizer import build_optimizer
from utils.edit_argparse import enforce_valid_defaults
from utils.train_eval_utils import (setup_training_dir,
                                    timers,
                                    write_final_eval_results,
                                    pigz_compress_tensorboard_file,
                                    record_postproc_time_table)

# specific to training this model
from models.neural_shared.neural_initializer import create_all_tstates 
from utils.edit_argparse import feedforward_fill_with_default_values as fill_with_default_values
from utils.edit_argparse import feedforward_share_top_level_args as share_top_level_args
from train_eval_fns.feedforward_predict_train_eval_one_batch import ( train_one_batch,
                                                                      eval_one_batch )
from train_eval_fns.neural_final_eval_wrapper import final_eval_wrapper
from train_eval_fns.TrainingWrapper import FeedforwardTrainingWrapper as TrainingWrapper


def train_feedforward(args, dataloader_dict: dict):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    err = (f"{args.pred_model_type} is not feedforward; "+
           f"using the wrong training script")
    assert args.pred_model_type == 'feedforward', err
    del err
    
    ### edit the argparse object in-place
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
    
    # create a new logfile
    with open(args.logfile_name,'w') as g:
        if not args.update_grads:
            g.write('DEBUG MODE: DISABLING GRAD UPDATES\n\n')
        
        g.write( f'Feedforward network to predict alignment-augmented descendant\n' )
        g.write( f'Ancestor sequence embedder (FULL-CONTEXT): {args.anc_model_type}\n' )
        g.write( f'Descendant sequence embedder (CAUSAL): {args.desc_model_type}\n' )
        g.write( f'Combine embeddings with: {args.pred_config["postproc_model_type"]}\n' )
        g.write( f'when reporting, normalizing losses by: {args.norm_reported_loss_by}\n\n' )
       
    
    ### save updated config, provide filename for saving model parameters
    encoder_save_model_filename = args.model_ckpts_dir + '/'+ f'ANC_ENC.pkl'
    decoder_save_model_filename = args.model_ckpts_dir + '/'+ f'DESC_DEC.pkl'
    finalpred_save_model_filename = args.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
    all_save_model_filenames = [encoder_save_model_filename, 
                                decoder_save_model_filename,
                                finalpred_save_model_filename]
    
    
    ### extract data from dataloader_dict
    training_dset = dataloader_dict['training_dset']
    training_dl = dataloader_dict['training_dl']
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    
    
    ###########################################################################
    ### 2: MODEL INIT, TRAINING  ##############################################
    ###########################################################################
    print('2: model init')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'2: model init\n')
    
    # init the optimizer
    tx = build_optimizer(args)
    rngkey, model_init_rngkey = jax.random.split(rngkey, num=2)
    
    
    ### determine shapes for init
    # unaligned sequences sizes
    global_seq_max_length = max([training_dset.global_seq_max_length,
                                 test_dset.global_seq_max_length])
    largest_seqs = (args.batch_size, global_seq_max_length)
    
    # aligned datasets sizes
    if args.use_scan_fns:
        max_dim1 = args.chunk_length
    
    elif not args.use_scan_fns:
        max_dim1 = max([training_dset.global_align_max_length,
                        test_dset.global_align_max_length]) - 1
      
    largest_aligns = (args.batch_size, max_dim1)
    del max_dim1
    
    seq_shapes = [largest_seqs, largest_aligns]
    
    # time
    t_per_sample = args.pred_config['t_per_sample']
    
    if t_per_sample:
        dummy_t_for_each_sample = jnp.empty( (args.batch_size,) )
    
    elif not t_per_sample:
        dummy_t_for_each_sample = None
    
    # batch provided to train/eval functions consist of:
    # 1.) unaligned sequences (B, L_seq, 2)
    # 2.) aligned data matrices (B, L_align, 4)
    # 3.) time per sample; (B,) if present, None otherwise
    # 4, not used.) sample index (B,)
    seq_shapes = [largest_seqs, largest_aligns, dummy_t_for_each_sample]
    
    
    ### initialize functions, determine concat_fn
    out = create_all_tstates( seq_shapes = seq_shapes, 
                              tx = tx, 
                              model_init_rngkey = model_init_rngkey,
                              tabulate_file_loc = args.model_ckpts_dir,
                              anc_model_type = args.anc_model_type, 
                              desc_model_type = args.desc_model_type, 
                              pred_model_type = args.pred_model_type, 
                              anc_enc_config = args.anc_enc_config, 
                              desc_dec_config = args.desc_dec_config, 
                              pred_config = args.pred_config,
                              t_array_for_all_samples = None )  
    all_trainstates, all_model_instances, concat_fn = out
    del out
    
    
    ### jit-compilations
    # training function
    parted_train_fn = partial( train_one_batch,
                                all_model_instances = all_model_instances,
                                interms_for_tboard = args.interms_for_tboard,
                                concat_fn = concat_fn,
                                norm_loss_by_for_reporting = args.norm_reported_loss_by,
                                update_grads = args.update_grads )
    
    train_fn_jitted = jax.jit(parted_train_fn, 
                              static_argnames = ['max_seq_len', 'max_align_len'])
    del parted_train_fn
    
    
    ### eval_fn used in training loop (to monitor progress)
    # pass arguments into eval_one_batch; make a parted_eval_fn that doesn't
    #   return any intermediates
    no_returns = {'encoder_sow_outputs': False,
                  'decoder_sow_outputs': False,
                  'finalpred_sow_outputs': False,
                  'gradients': False,
                  'weights': False,
                  'optimizer': False,
                  'ancestor_embeddings': False,
                  'descendant_embeddings': False,
                  'forward_pass_outputs': False,
                  'final_logprobs': False}
    extra_args_for_eval = dict()
    
    # if this is a transformer model, will have extra arguments for eval funciton
    if (args.anc_model_type == 'Transformer' or args.desc_model_type == 'Transformer'):
        extra_args_for_eval['output_attn_weights'] = False
    
    parted_eval_fn = partial( eval_one_batch,
                              all_model_instances = all_model_instances,
                              interms_for_tboard = no_returns,
                              concat_fn = concat_fn,
                              norm_loss_by_for_reporting = args.norm_reported_loss_by,
                              extra_args_for_eval = extra_args_for_eval )
    
    # jit compile this eval function
    eval_fn_jitted = jax.jit(parted_eval_fn, 
                              static_argnames = ['max_seq_len', 'max_align_len'])
    del parted_eval_fn
    
    
    ### initialize training wrapper
    training_wrapper = TrainingWrapper( args = args,
                                        epoch_arr = range(args.num_epochs),
                                        initial_training_rngkey = rngkey,
                                        dataloader_dict = dataloader_dict,
                                        train_fn_jitted = train_fn_jitted,
                                        eval_fn_jitted = eval_fn_jitted,
                                        all_save_model_filenames = all_save_model_filenames,
                                        writer = writer )
    
    
    ### train
    print(f'3: main training loop')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'3: main training loop\n')
    
    out = training_wrapper.run_train_loop( all_trainstates = all_trainstates )
    early_stop, best_epoch, best_trainstates = out
    del out
    
    
    
    ###########################################################################
    ### FINAL EVAL   ##########################################################
    ###########################################################################
    print(f'4: post-training actions')
    # write to logfile
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'4: post-training actions\n')
    
    # don't accidentally use old trainstates or eval fn
    del all_trainstates, eval_fn_jitted
    
    # new timer
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


    ### jit compile new eval function
    # if this is a transformer model, will have extra arguments for eval funciton
    extra_args_for_eval = dict()
    if (args.anc_model_type == 'transformer' and args.desc_model_type == 'transformer'):
        flag = (args.anc_enc_config.get('output_attn_weights',False) or 
                args.desc_dec_config.get('output_attn_weights',False))
        extra_args_for_eval['output_attn_weights'] = flag

    parted_eval_fn = partial( eval_one_batch,
                              all_model_instances = all_model_instances,
                              interms_for_tboard = args.interms_for_tboard,
                              concat_fn = concat_fn,
                              norm_loss_by_for_reporting = args.norm_reported_loss_by,  
                              extra_args_for_eval = extra_args_for_eval )
    del extra_args_for_eval

    # jit compile this eval function
    eval_fn_jitted = jax.jit( parted_eval_fn, 
                              static_argnames = ['max_seq_len', 'max_align_len'])
    del parted_eval_fn

    ###########################################
    ### loop through training dataloader and  #
    ### score with best params                #
    ###########################################
    with open(args.logfile_name,'a') as g:
        g.write(f'SCORING ALL TRAIN SEQS\n')
        
    train_summary_stats = final_eval_wrapper(dataloader = training_dl, 
                                             dataset = training_dset, 
                                             best_trainstates = best_trainstates, 
                                             jitted_determine_seqlen_bin = training_wrapper.seqlen_bin_fn,
                                             jitted_determine_alignlen_bin = training_wrapper.alignlen_bin_fn,
                                             eval_fn_jitted = eval_fn_jitted,
                                             out_alph_size = args.out_alph_size,
                                             save_arrs = args.save_arrs,
                                             save_per_sample_losses = args.save_per_sample_losses,
                                             interms_for_tboard = args.interms_for_tboard, 
                                             logfile_dir = args.logfile_dir,
                                             out_arrs_dir = args.out_arrs_dir,
                                             outfile_prefix = f'train-set',
                                             tboard_writer = writer)


    ###########################################
    ### loop through test dataloader and      #
    ### score with best params                #
    ###########################################
    with open(args.logfile_name,'a') as g:
        g.write(f'SCORING ALL TEST SEQS\n')
        
    # output_attn_weights also controlled by cond1 and cond2
    test_summary_stats = final_eval_wrapper(dataloader = test_dl, 
                                             dataset = test_dset, 
                                             best_trainstates = best_trainstates, 
                                             jitted_determine_seqlen_bin = training_wrapper.seqlen_bin_fn,
                                             jitted_determine_alignlen_bin = training_wrapper.alignlen_bin_fn,
                                             eval_fn_jitted = eval_fn_jitted,
                                             out_alph_size = args.out_alph_size, 
                                             save_arrs = args.save_arrs,
                                             save_per_sample_losses = args.save_per_sample_losses,
                                             interms_for_tboard = args.interms_for_tboard, 
                                             logfile_dir = args.logfile_dir,
                                             out_arrs_dir = args.out_arrs_dir,
                                             outfile_prefix = f'test-set',
                                             tboard_writer = writer)


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

    # clean up intermediates
    for file_path in glob.glob(f"{args.model_ckpts_dir}/*_INPROGRESS.pkl"):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass  # File might have been deleted already