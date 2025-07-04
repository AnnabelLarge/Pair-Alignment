#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 14:24:59 2025

@author: annabel
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

# jax/flax stuff
import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import optax

# pytorch imports
from torch.utils.data import DataLoader

# custom function/classes imports
from train_eval_fns.build_optimizer import build_optimizer
from utils.sequence_length_helpers import (determine_seqlen_bin, 
                                           determine_alignlen_bin)
from utils.write_timing_file import write_timing_file
from utils.write_approx_dict import write_approx_dict
from utils.edit_argparse import (enforce_valid_defaults,
                                 fill_with_default_values,
                                 share_top_level_args)

# specific to training this model
from models.neural_utils.neural_initializer import create_all_tstates 
from train_eval_fns.neural_tkf_train_eval import eval_one_batch
from train_eval_fns.full_length_final_eval_wrapper import final_eval_wrapper



def eval_neural_hmm( args, 
                     dataloader_dict: dict,
                     training_argparse ):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    err = (f"{training_argparse.pred_model_type} is not neural_hmm; "+
           f"using the wrong eval script")
    assert training_argparse.pred_model_type == 'neural_hmm', err
    del err
    
    prev_model_ckpts_dir = f'{os.getcwd()}/{args.training_wkdir}/model_ckpts'
    pairhmm_savemodel_filename = prev_model_ckpts_dir + '/'+ f'FINAL_PRED_BEST.pkl'
    
    fill_with_default_values(training_wkdir)
    enforce_valid_defaults(training_wkdir)
    share_top_level_args(training_wkdir)
    
    
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
        g.write( f'Loading from {args.training_wkdir} to eval new data\n\n' )
            
        g.write(f'Neural sequence embedders with Markovian alignment assumption\n')
        g.write( f'Substitution model: {training_argparse.pred_config["subst_model_type"]}\n' )
        g.write( f'Indel model: {training_argparse.pred_config["indel_model_type"]}\n' )
        g.write( f'when reporting, normalizing losses by: {training_argparse.norm_loss_by}\n\n' )
        
        g.write( f'Evolutionary model parameters (global vs local):\n' )
        
        if not training_argparse.pred_config['load_all']:
            for key, val in training_argparse.pred_config["global_or_local"].items():
                g.write(f'{key}: {val}\n')
                
        g.write('\n')
        g.write(f'Ancestor sequence embedder (FULL-CONTEXT): {training_argparse.anc_model_type}\n')
        g.write(f'Descendant sequence embedder (CAUSAL): {training_argparse.desc_model_type}\n\n')
        
        
    ### provide filenames of saved model parameters
    encoder_save_model_filename = training_argparse.model_ckpts_dir + '/'+ f'ANC_ENC.pkl'
    decoder_save_model_filename = training_argparse.model_ckpts_dir + '/'+ f'DESC_DEC.pkl'
    finalpred_save_model_filename = training_argparse.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
    all_save_model_filenames = [encoder_save_model_filename, 
                                decoder_save_model_filename,
                                finalpred_save_model_filename]
    
    
    ### extract data from dataloader_dict
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    t_array_for_all_samples = dataloader_dict['t_array_for_all_samples']
    
    
    ###########################################################################
    ### 2: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
    ###########################################################################
    print('MODEL INIT')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'1: model init\n')
    
    # need to intialize an optimizer for compatibility when restoring the state, 
    #   but we're not training so this doesn't really matter?
    tx = build_optimizer(training_argparse)
    
    
    ### determine shapes for init
    # unaligned sequences sizes
    global_seq_max_length = test_dset.global_seq_max_length
    largest_seqs = (args.batch_size, global_seq_max_length)
    
    # aligned datasets sizes
    if args.use_scan_fns:
        max_dim1 = args.chunk_length
    
    elif not args.use_scan_fns:
        max_dim1 = test_dset.global_align_max_length - 1
      
    largest_aligns = (args.batch_size, max_dim1)
    del max_dim1
    
    # time
    if t_array_for_all_samples is not None:
        dummy_t_array_for_all_samples = jnp.empty( (t_array_for_all_samples.shape[0], ) )
        dummy_t_for_each_sample = None
    
    else:
        dummy_t_array_for_all_samples = None
        dummy_t_for_each_sample = jnp.empty( (args.batch_size,) )
    
    # batch provided to train/eval functions consist of:
    # 1.) unaligned sequences (B, L_seq, 2)
    # 2.) aligned data matrices (B, L_align, 5)
    # 3.) time per sample (if applicable) (B,)
    # 4, not used.) sample index (B,)
    seq_shapes = [largest_seqs, largest_aligns, dummy_t_for_each_sample]
    
    
    ### initialize trainstate objects, concat_fn
    out = create_all_tstates( seq_shapes = seq_shapes, 
                              tx = tx, 
                              model_init_rngkey = jax.random.key(0),
                              tabulate_file_loc = args.model_ckpts_dir,
                              anc_model_type = training_argparse.anc_model_type, 
                              desc_model_type = training_argparse.desc_model_type, 
                              pred_model_type = training_argparse.pred_model_type, 
                              anc_enc_config = training_argparse.anc_enc_config, 
                              desc_dec_config = training_argparse.desc_dec_config, 
                              pred_config = training_argparse.pred_config,
                              t_array_for_all_samples = dummy_t_array_for_all_samples,
                              )  
    blank_trainstates, all_model_instances, concat_fn = out
    del out
    
    # load parameters
    best_trainstates = []
    for i in range(3):
        param_fname = all_save_model_filenames[i]
        blank_tstate = blank_trainstates[i]
        with open(f'{training_argparse.model_ckpts_dir}/{param_fname}', 'rb') as f:
            state_dict = pickle.load(f)
        ts = flax.serialization.from_state_dict( blank_tstate, state_dict )
        best_trainstates.append(ts)
        del param_fname, blank_tstate, f, state_dict, ts
    
    del i, blank_trainstates
    
    
    ### jit-compilations
    # helpers to determine when to jit-compile (according to seq/align length 
    #   combination)
    parted_determine_alignlen_bin = partial(determine_alignlen_bin,  
                                            chunk_length = args.chunk_length,
                                            seq_padding_idx = args.seq_padding_idx)
    jitted_determine_alignlen_bin = jax.jit(parted_determine_alignlen_bin)
    del parted_determine_alignlen_bin
    
    parted_determine_seqlen_bin = partial(determine_seqlen_bin,
                                          chunk_length = args.chunk_length, 
                                          seq_padding_idx = args.seq_padding_idx)
    jitted_determine_seqlen_bin = jax.jit(parted_determine_seqlen_bin)
    del parted_determine_seqlen_bin
    
    ### eval_fn used in training loop (to monitor progress)
    # pass arguments into eval_one_batch; make a parted_eval_fn that doesn't
    #   return any intermediates
    no_returns = {'encoder_sow_outputs': False,
                  'decoder_sow_outputs': False,
                  'finalpred_sow_outputs': False,
                  'gradients': False,
                  'weights': False,
                  'ancestor_embeddings': False,
                  'descendant_embeddings': False,
                  'forward_pass_outputs': args.save_scoremats}
    extra_args_for_eval = dict()
    
    # if this is a transformer model, will have extra arguments for eval funciton
    if (args.anc_model_type == 'Transformer' or args.desc_model_type == 'Transformer'):
        extra_args_for_eval['output_attn_weights'] = False
    
    parted_eval_fn = partial( eval_one_batch,
                              all_model_instances = all_model_instances,
                              interms_for_tboard = no_returns,
                              t_array_for_all_samples = t_array_for_all_samples,  
                              concat_fn = concat_fn,
                              norm_loss_by_for_reporting = args.norm_loss_by,                  
                              extra_args_for_eval = extra_args_for_eval )
    del no_returns, extra_args_for_eval
    
    # jit compile this eval function
    eval_fn_jitted = jax.jit( parted_eval_fn, 
                              static_argnames = ['max_seq_len',
                                                 'max_align_len'])
    del parted_eval_fn

    
    ###########################################################################
    ### 3: EVAL   #############################################################
    ###########################################################################
    print(f'BEGIN EVAL')
    # write to logfile
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'BEGIN EVAL\n')
        
    test_summary_stats = final_eval_wrapper(dataloader = test_dl, 
                                             dataset = test_dset, 
                                             best_trainstates = best_trainstates, 
                                             jitted_determine_seqlen_bin = jitted_determine_seqlen_bin,
                                             jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                                             eval_fn_jitted = eval_fn_jitted,
                                             out_alph_size = args.full_alphabet_size, 
                                             save_arrs = args.save_scoremats,
                                             save_per_sample_losses = args.save_per_sample_losses,
                                             interms_for_tboard = args.interms_for_tboard, 
                                             logfile_dir = args.logfile_dir,
                                             out_arrs_dir = args.out_arrs_dir,
                                             outfile_prefix = f'dset',
                                             tboard_writer = None)
    
    ###########################################
    ### update the logfile with final losses  #
    ###########################################
    # save the trainstate again
    for i in range(3):
        param_fname = all_save_model_filenames[i]
        with open(f'{args.model_ckpts_dir}/{param_fname}', 'wb') as g:
            model_state_dict = flax.serialization.to_state_dict(best_trainstates[i])
            pickle.dump(model_state_dict, g)
        
    to_write = {'RUN': args.eval_wkdir}
    to_write = {**to_write, **test_summary_stats}
    
    with open(f'{args.logfile_dir}/AVE-LOSSES.tsv','w') as g:
        for k, v in to_write.items():
            g.write(f'{k}\t{v}\n')
