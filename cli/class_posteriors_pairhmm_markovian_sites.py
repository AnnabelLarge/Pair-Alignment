#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 21:13:38 2025

@author: annabel_large
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
from train_eval_fns.markovian_site_classes_training_fns import ( label_class_posteriors )


def class_posteriors_pairhmm_markovian_sites( args, 
                                              dataloader_dict: dict,
                                              training_argparse ):
    """
    after training with SGD, post-hoc label class marginal posteriors
      per class AND PER TIME
    
    place this back in the original directory!
    """
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


    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    ### create the eval working directory, if it doesn't exist
    args.out_arrs_dir = f'{args.training_wkdir}/class_marginals'    
    if 'class_marginals' not in os.listdir(args.training_wkdir):
        os.mkdir(args.out_arrs_dir)
    
    
    ### extract data from dataloader_dict
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    
    
    ###########################################################################
    ### 1: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
    ###########################################################################
    print('1: model init')
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
    
    
    ### part+jit label_class_posteriors function
    t_array = test_dset.return_time_array()
    
    
    
    parted_eval_fn = partial( label_class_posteriors,
                              t_array = t_array,
                              pairhmm_trainstate = best_pairhmm_trainstate,
                              pairhmm_instance = pairhmm_instance )
    
    eval_fn_jitted = jax.jit( parted_eval_fn, 
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
    for batch_idx, batch in tqdm( enumerate(test_dl), total=len(test_dl) ): 
        batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch)
        batch_max_alignlen = batch_max_alignlen.item()
        
        marginals = eval_fn_jitted( batch=batch, 
                                    max_align_len=batch_max_alignlen )
        
        with open(f'{out_arrs_dir}/dset-pt-{batch_idx}_class_posterior_marginals.npy', 'wb') as g:
            np.save(g, marginals)
        
        label_df = test_dset.retrieve_sample_names(batch[-1])
        label_df.to_csv('{out_arrs_dir}/dset-pt-{batch_idx}_ROW-LABELS_class_posterior_marginals.npy', sep='\t', index_col=0)
        
