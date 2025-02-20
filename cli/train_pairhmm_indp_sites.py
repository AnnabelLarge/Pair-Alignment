#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:33:01 2025

@author: annabel

Load parameters and evaluate likelihoods for an independent
  site class model

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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# custom function/classes imports (in order of appearance)
from train_eval_fns.build_optimizer import build_optimizer
from utils.write_config import write_config
from utils.edit_argparse import (enforce_valid_defaults,
                                 fill_with_default_values,
                                 share_top_level_args)
from utils.setup_training_dir import setup_training_dir
from utils.sequence_length_helpers import (determine_seqlen_bin, 
                                           determine_alignlen_bin)
from utils.tensorboard_recording_utils import (write_times,
                                               write_optional_outputs_during_training)
from utils.write_timing_file import write_timing_file

# specific to training this model
from models.simple_site_class_predict.initializers import init_pairhmm_indp_sites as init_pairhmm
from train_eval_fns.indp_site_classes_training_fns import ( train_one_batch,
                                                            eval_one_batch,
                                                            final_eval_wrapper )

def bounded_sigmoid(x, min_val, max_val):
    return min_val + (max_val - min_val) / (1 + jnp.exp(-x))
    


def train_pairhmm_indp_sites(args, dataloader_dict: dict):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    err = (f"{args.pred_model_type} is not pairhmm_indp_sites; "+
           f"using the wrong training script")
    assert args.pred_model_type == 'pairhmm_indp_sites', err
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
            
        g.write( (f'PairHMM {args.pred_config["indel_model_type"]} '+
                  f'with independent site classes over emissions\n')
                )
        g.write( (f'  - Number of site classes for substitution model: '+
                  f'{args.pred_config["num_emit_site_classes"]}\n' )
                )
        g.write(f'  - Loss function: {args.loss_type}\n')
        g.write(f'  - Normalizing losses by: {args.norm_loss_by}\n')
    
    
    ### save updated config, provide filename for saving model parameters
    finalpred_save_model_filename = args.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
    write_config(args = args, out_dir = args.model_ckpts_dir)
    
    
    ### extract data from dataloader_dict
    training_dset = dataloader_dict['training_dset']
    training_dl = dataloader_dict['training_dl']
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    args.pred_config['training_dset_aa_counts'] = training_dset.AAcounts
    
    
    
    ###########################################################################
    ### 2: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
    ###########################################################################
    print('2: model init')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'2: model init\n')
    
    
    # init the optimizer, split a new rng key
    tx = build_optimizer(args)
    rngkey, model_init_rngkey = jax.random.split(rngkey, num=2)
    
    
    ### determine shapes for init
    # time
    num_timepoints = test_dset.retrieve_num_timepoints(times_from = args.pred_config['times_from'])
    dummy_t_array = jnp.empty( (num_timepoints, ) )
    
    # counts array
    B = args.batch_size
    alph = args.emission_alphabet_size
    num_states = training_dset.num_transitions
    dummy_subCounts = jnp.empty( (B, alph, alph) )
    dummy_insCounts = jnp.empty( (B, alph) )
    dummy_delCounts = jnp.empty( (B, alph) )
    dummy_transCounts = jnp.empty( (B, num_states, num_states) )
    
    seq_shapes = [dummy_subCounts,
                  dummy_insCounts,
                  dummy_delCounts,
                  dummy_transCounts]
    
    
    ### initialize functions
    out = init_pairhmm( seq_shapes = seq_shapes, 
                        dummy_t_array = dummy_t_array,
                        tx = tx, 
                        model_init_rngkey = model_init_rngkey,
                        pred_config = args.pred_config,
                        tabulate_file_loc = args.model_ckpts_dir
                        )
    all_trainstates, all_model_instances = out
    del out
    
    ### part+jit training function
    # note: if you want to use a different time per sample, will
    #  have to change this jit compilation
    t_array = test_dset.return_time_array()
    parted_train_fn = partial( train_one_batch,
                               interms_for_tboard = args.interms_for_tboard,
                               update_grads = args.update_grads,
                               t_array = t_array )
    
    train_fn_jitted = jax.jit(parted_train_fn)
    del parted_train_fn
    
    
    ### part+jit eval function
    # note: if you want to use a different time per sample, will
    #  have to change this jit compilation
    parted_eval_fn = partial( eval_one_batch,
                               interms_for_tboard = {'finalpred_sow_outputs': False},
                               t_array = t_array )
    eval_fn_jitted = jax.jit(parted_eval_fn)
    del parted_eval_fn
    
    
    
    ###########################################################################
    ### 3: START TRAINING LOOP   ##############################################
    ###########################################################################
    print(f'3: main training loop')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'3: main training loop\n')
    
    # when to save/what to save
    best_epoch = -1
    best_test_loss = 999999
    best_trainstates = all_trainstates
    
    # quit training if test loss increases for X epochs in a row
    prev_test_loss = 999999
    early_stopping_counter = 0
    
    # rng key for train
    rngkey, training_rngkey = jax.random.split(rngkey, num=2)
    
    # record time spent at each phase (use numpy array to store)
    all_train_set_times = np.zeros( (args.num_epochs,2) )
    all_eval_set_times = np.zeros( (args.num_epochs,2) )
    all_epoch_times = np.zeros( (args.num_epochs,2) )
    
    for epoch_idx in tqdm(range(args.num_epochs)):
        epoch_real_start = wall_clock_time()
        epoch_cpu_start = process_time()
        
        ave_epoch_train_loss = 0
        ave_epoch_train_perpl = 0
        
#__4___8: epoch level (two tabs)          
        ##############################################
        ### 3.1: train and update model parameters   #
        ##############################################
        train_real_start = wall_clock_time()
        train_cpu_start = process_time()
        
        for batch_idx, batch in enumerate(training_dl):   
            batch_epoch_idx = epoch_idx * len(training_dl) + batch_idx  
            
            rngkey_for_training_batch = jax.random.fold_in(training_rngkey, epoch_idx+batch_idx)
            out = train_fn_jitted(batch=batch, 
                                  training_rngkey=rngkey_for_training_batch, 
                                  all_trainstates=all_trainstates)
            train_metrics, all_trainstates = out
            del out
        

#__4___8__12: batch level (three tabs)
            ################################################################
            ### 3.2: if NaN is found, save current progress for inspection #
            ###      and quit training                                     #
            ################################################################
            if jnp.isnan( train_metrics['batch_loss'] ):
                with open(args.logfile_name,'a') as g:
                    g.write('\n')
                    g.write(f'NaN loss at epoch {epoch_idx}, batch {batch_idx}\n')
                    
                # save the argparse object by itself
                args.epoch_idx = epoch_idx
                with open(f'{args.model_ckpts_dir}/TRAINING_ARGPARSE.pkl', 'wb') as g:
                    pickle.dump(args, g)
                
                # save all trainstate objects
                new_outfile = finalpred_save_model_filename.replace('.pkl','_BROKEN.pkl')
                with open(new_outfile, 'wb') as g:
                    model_state_dict = flax.serialization.to_state_dict(all_trainstates)
                    pickle.dump(model_state_dict, g)
                
                raise RuntimeError( ('NaN loss detected; saved intermediates '+
                                    'and quit training') )
            
            
#__4___8__12: batch level (three tabs)
            ### add to recorded metrics for this epoch
            weight = args.batch_size / len(training_dset)
            ave_epoch_train_loss += train_metrics['batch_loss'] * weight
            ave_epoch_train_perpl += train_metrics['batch_ave_perpl'] * weight
            del weight
            
            # record metrics
            interm_rec = batch_epoch_idx % args.histogram_output_freq == 0
            final_rec = (batch_idx == len(training_dl)) & (epoch_idx == args.num_epochs)
            
            write_optional_outputs_during_training(writer_obj = writer, 
                                                    all_trainstates = all_trainstates,
                                                    global_step = batch_epoch_idx, 
                                                    dict_of_values = train_metrics, 
                                                    interms_for_tboard = args.interms_for_tboard, 
                                                    write_histograms_flag = interm_rec or final_rec)
            
            
#__4___8: epoch level (two tabs)
        ### manage timing
        # stop timer
        train_real_end = wall_clock_time()
        train_cpu_end = process_time()

        # record the CPU+system and wall-clock (real) time
        write_times(cpu_start = train_cpu_start, 
                    cpu_end = train_cpu_end, 
                    real_start = train_real_start, 
                    real_end = train_real_end, 
                    tag = 'Process training data', 
                    step = epoch_idx, 
                    writer_obj = writer)
        
        # also record for later
        all_train_set_times[epoch_idx, 0] = train_real_end - train_real_start
        all_train_set_times[epoch_idx, 1] = train_cpu_end - train_cpu_start
        
        del train_cpu_start, train_cpu_end
        del train_real_start, train_real_end
        
        
        ##############################################################
        ### 3.3: also check current performance on held-out test set #
        ##############################################################
        # Note: it's possible to output intermediates for these points too;
        # but right now, that's not collected
        ave_epoch_test_loss = 0
        ave_epoch_test_perpl = 0
        
        # start timer
        eval_real_start = wall_clock_time()
        eval_cpu_start = process_time()
        
        for batch_idx, batch in enumerate(test_dl):
            eval_metrics = eval_fn_jitted(batch=batch, 
                                          all_trainstates=all_trainstates)
            
            ### add to total loss for this epoch; weight by number of
            ###   samples/valid tokens in this batch
            weight = args.batch_size / len(test_dset)
            ave_epoch_test_loss += eval_metrics['batch_loss'] * weight
            ave_epoch_test_perpl += jnp.mean( eval_metrics['perplexity_perSamp'] ) * weight
            del weight
    
    
#__4___8: epoch level (two tabs)
        ### manage timing
        # stop timer
        eval_real_end = wall_clock_time()
        eval_cpu_end = process_time()

        # record the CPU+system and wall-clock (real) time to tensorboard
        write_times(cpu_start = eval_cpu_start, 
                    cpu_end = eval_cpu_end, 
                    real_start = eval_real_start, 
                    real_end = eval_real_end, 
                    tag = 'Process test set data', 
                    step = epoch_idx, 
                    writer_obj = writer)
        
        # also record for later
        all_eval_set_times[epoch_idx, 0] = eval_real_end - eval_real_start
        all_eval_set_times[epoch_idx, 1] = eval_cpu_end - eval_cpu_start
        
        del eval_cpu_start, eval_cpu_end
        del eval_real_start, eval_real_end
        
        
        ##########################################
        ### 3.4: record scalars to tensorboard   #
        ##########################################
        # training set
        writer.add_scalar(tag ='Loss/training set', 
                          scalar_value = ave_epoch_train_loss.item(), 
                          global_step = epoch_idx)
        
        writer.add_scalar(tag='Perplexity/training set',
                          scalar_value=ave_epoch_train_perpl.item(), 
                          global_step=epoch_idx)
        
        # test set
        writer.add_scalar(tag='Loss/test set', 
                          scalar_value=ave_epoch_test_loss.item(), 
                          global_step=epoch_idx)
        
        writer.add_scalar(tag='Perplexity/test set',
                          scalar_value=ave_epoch_test_perpl.item(), 
                          global_step=epoch_idx)
        
        
#__4___8: epoch level (two tabs) 
        ##########################################################
        ### 3.5: if this is the best epoch TEST loss,            #
        ###      save the model params and args for later eval   #
        ##########################################################
        if ave_epoch_test_loss < best_test_loss:
            with open(args.logfile_name,'a') as g:
                g.write((f'New best test loss at epoch {epoch_idx}: ') +
                        (f'{ave_epoch_test_loss}\n'))
            
            # update "best" recordings
            best_test_loss = ave_epoch_test_loss
            best_trainstates = all_trainstates
            best_epoch = epoch_idx
            
            # save models to regular python pickles too (in case training is 
            #   interrupted)
            with open(finalpred_save_model_filename, 'wb') as g:
                model_state_dict = flax.serialization.to_state_dict(all_trainstates)
                pickle.dump(model_state_dict, g)
            
            
#__4___8: epoch level (two tabs) 
        ###########################
        ### 3.6: EARLY STOPPING   #
        ###########################
        ### condition 1: if test loss stagnates or starts to go up, compared
        ###              to previous epoch's test loss
        cond1 = jnp.allclose (prev_test_loss, 
                              jnp.minimum (prev_test_loss, ave_epoch_test_loss), 
                              atol=args.early_stop_cond1_atol)

        ### condition 2: if test loss is substatially worse than best test loss
        cond2 = (ave_epoch_test_loss - best_test_loss) > args.early_stop_cond2_gap

        if cond1 or cond2:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        
        if early_stopping_counter == args.patience:
            # record in the raw ascii logfile
            with open(args.logfile_name,'a') as g:
                g.write(f'\n\nEARLY STOPPING AT {epoch_idx}:\n')
            
            # record time spent at this epoch
            epoch_real_end = wall_clock_time()
            epoch_cpu_end = process_time()
            all_epoch_times[epoch_idx, 0] = epoch_real_end - epoch_real_start
            all_epoch_times[epoch_idx, 1] = epoch_cpu_end - epoch_cpu_start
            
            write_times(cpu_start = epoch_cpu_start, 
                        cpu_end = epoch_cpu_end, 
                        real_start = epoch_real_start, 
                        real_end = epoch_real_end, 
                        tag = 'Process one epoch', 
                        step = epoch_idx, 
                        writer_obj = writer)
            
            del epoch_cpu_start, epoch_cpu_end
            del epoch_real_start, epoch_real_end
            
            # save the trainstates for later use
            best_trainstates = all_trainstates
            
            # rage quit
            break


        ### before next epoch, remember this epoch's loss for next iteration
        prev_test_loss = ave_epoch_test_loss
        
        # record time spent at this epoch
        epoch_real_end = wall_clock_time()
        epoch_cpu_end = process_time()
        all_epoch_times[epoch_idx, 0] = epoch_real_end - epoch_real_start
        all_epoch_times[epoch_idx, 1] = epoch_cpu_end - epoch_cpu_start
        
        # write to tensorboard
        write_times(cpu_start = epoch_cpu_start, 
                    cpu_end = epoch_cpu_end, 
                    real_start = epoch_real_start, 
                    real_end = epoch_real_end, 
                    tag = 'Process one epoch', 
                    step = epoch_idx, 
                    writer_obj = writer)
        
        del epoch_cpu_start, epoch_cpu_end, epoch_real_start, epoch_real_end
        
        

    ###########################################################################
    ### 4: POST-TRAINING ACTIONS   ############################################
    ###########################################################################
    print(f'4: post-training actions')
    # write to logfile
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'4: post-training actions\n')
    
    # don't accidentally use old trainstates or eval fn
    del all_trainstates, eval_fn_jitted
    
    
    ### handle time
    # write final timing
    write_timing_file( outdir = args.logfile_dir,
                       train_times = all_train_set_times,
                       eval_times = all_eval_set_times,
                       total_times = all_epoch_times )
    
    del all_train_set_times, all_eval_set_times, all_epoch_times

    # new timer
    post_training_real_start = wall_clock_time()
    post_training_cpu_start = process_time()
    
    
    ### write to output logfile
    with open(args.logfile_name,'a') as g:
        # if early stopping was never triggered, record results at last epoch
        if early_stopping_counter != args.patience:
            g.write(f'Regular stopping after {epoch_idx} full epochs:\n\n')
        
        # finish up logfile, regardless of early stopping or not
        g.write(f'Epoch with lowest average test loss ("best epoch"): {best_epoch}\n')
        g.write(f'RE-EVALUATING ALL DATA WITH BEST PARAMS:\n\n')
    
    del epoch_idx
    
    
    ### un-transform parameters and write to numpy arrays
    all_model_instances.write_params(tstate = best_trainstates,
                                     out_folder = args.out_arrs_dir,
                                     pred_config = args.pred_config)
    
    
    ### jit-compile new eval function
    t_array = test_dset.return_time_array()
    parted_eval_fn = partial( eval_one_batch,
                               interms_for_tboard = args.interms_for_tboard,
                               t_array = t_array )
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
                                             best_trainstates = best_trainstates, 
                                             eval_fn_jitted = eval_fn_jitted,
                                             save_per_sample_losses = args.save_per_sample_losses,
                                             logfile_dir = args.logfile_dir,
                                             out_arrs_dir = args.out_arrs_dir,
                                             outfile_prefix = f'train-set')


    ###########################################
    ### loop through training dataloader and  #
    ### score with best params                #
    ###########################################
    with open(args.logfile_name,'a') as g:
        g.write(f'SCORING ALL TEST SEQS\n\n')
        
    test_summary_stats = final_eval_wrapper(dataloader = test_dl, 
                                            dataset = test_dset, 
                                            best_trainstates = best_trainstates, 
                                            eval_fn_jitted = eval_fn_jitted,
                                            save_per_sample_losses = args.save_per_sample_losses,
                                            logfile_dir = args.logfile_dir,
                                            out_arrs_dir = args.out_arrs_dir,
                                            outfile_prefix = f'test-set')
    
    
    ###########################################
    ### update the logfile with final losses  #
    ###########################################
    to_write = {'RUN': args.training_wkdir,
                'train_ave_{args.loss_type}_loss_seqlen_normed': train_summary_stats['final_ave_loss_seqlen_normed'],
                'train_perplexity': train_summary_stats['final_perplexity'],
                'train_ece': train_summary_stats['final_ece'] ,
                'test_ave_{args.loss_type}_loss_seqlen_normed': test_summary_stats['final_ave_loss_seqlen_normed'],
                'test_perplexity': test_summary_stats['final_perplexity'],
                'test_ece': test_summary_stats['final_ece']
                }
    
    with open(f'{args.logfile_dir}/AVE-LOSSES.tsv','w') as g:
        for k, v in to_write.items():
            g.write(f'{k}\t{v}\n')
    
    post_training_real_end = wall_clock_time()
    post_training_cpu_end = process_time()
    
    # record total time spent on post-training actions; write this to a table
    #   instead of a scalar
    cpu_sys_time = post_training_cpu_end - post_training_cpu_start
    real_time = post_training_real_end - post_training_real_start
    
    df = pd.DataFrame({'label': ['CPU+sys time', 'Real time'],
                       'value': [cpu_sys_time, real_time]})
    markdown_table = df.to_markdown()
    writer.add_text(tag = 'Code Timing | Post-training actions',
                    text_string = markdown_table,
                    global_step = 0)
    
    # when you're done with the function, close the tensorboard writer and
    #   compress the output file
    writer.close()
    
    # don't remove source on macOS (when I'm doing CPU testing)
    print('\n\nDONE; compressing tboard folder')
    if platform.system() == 'Darwin':
        os.system(f"tar -czvf {args.training_wkdir}/tboard.tar.gz {args.training_wkdir}/tboard")
    
    # DO remove source on linux (when I'm doing real experiments)
    elif platform.system() == 'Linux':
        os.system(f"tar -czvf {args.training_wkdir}/tboard.tar.gz  --remove-files {args.training_wkdir}/tboard")
    
    
    