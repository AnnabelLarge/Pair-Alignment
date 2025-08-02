#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 17:42:19 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import numpy 
import pickle
import sys
from tqdm import tqdm







from utils.train_eval_utils import ( timers,
                                     metrics_for_epoch,
                                     write_timing_file,
                                     best_models_base_class,
                                     write_approx_dict,
                                     jit_compilation_tracker )
from utils.tensorboard_recording_utils import (write_optional_outputs_during_training)

from models.neural_shared.save_all_neural_trainstates import save_all_neural_trainstates


###############################################################################
### Helpers specific to the neural training loop   ############################
###############################################################################
class _best_models_neural( best_models_base_class ):
    def maybe_save_best_model(self,
                         args,
                         epoch_loss,
                         epoch_idx,
                         all_trainstates,
                         all_save_model_filenames):
        if epoch_loss < self.best_test_loss:
            with open(args.logfile_name,'a') as g:
                g.write( f'New best test loss at epoch {epoch_idx}: {epoch_loss}\n' )
            
            # update "best" recordings
            self.best_test_loss = epoch_loss
            self.best_trainstates = all_trainstates
            self.best_epoch = epoch_idx
            
            # save models to regular python pickles too (in case training is 
            #   interrupted)
            save_all_neural_trainstates( all_save_model_filenames = all_save_model_filenames,
                                  all_trainstates = all_trainstates, 
                                  suffix = 'BEST' )
    
def _set_sequence_lengths_for_jit( args,
                                   determine_seqlen_bin_fn,
                                   determine_alignlen_bin,
                                   batch ):
    # unpack briefly to get max len and number of samples in the 
    #   batch; place in some bin (this controls how many jit 
    #   compilations you do)
    batch_max_seqlen = determine_seqlen_bin_fn(batch = batch).item()
    batch_max_alignlen = determine_alignlen_bin(batch = batch).item()
    
    # I've had so much trouble with this ugh
    if args.use_scan_fns:
        err = (f'batch_max_alignlen (not including bos) is: '+
               f'{batch_max_alignlen - 1}'+
               f', which is not divisible by length for scan '+
               f'({args.chunk_length})')
        assert (batch_max_alignlen - 1) % args.chunk_length == 0, err
    
    return batch_max_seqlen, batch_max_alignlen


def _maybe_checkpoint_neural_tstates_during_training( args,
                                      prev_checkpoint_counter,
                                      epoch_idx,
                                      batch_idx,
                                      batch_loss,
                                      all_trainstates,
                                      all_save_model_filenames ):
    new_counter = prev_checkpoint_counter + 1
    
    cond1 = args.checkpoint_freq_during_training > 0
    cond2 = new_counter % args.checkpoint_freq_during_training == 0
    
    if cond1 and cond2:
        # save some metadata about the trainstate files
        with open(f'{args.model_ckpts_dir}/INPROGRESS_trainstates_info.txt','w') as g:
            g.write(f'Checkpoint created at: epoch {epoch_idx}, batch {batch_idx}\n')
            g.write(f'Current loss for the training set batch is: {batch_loss}\n')
        
        # save the trainstates
        save_all_neural_trainstates( all_save_model_filenames = all_save_model_filenames,
                              all_trainstates = all_trainstates, 
                              suffix = 'INPROGRESS' )
        
        # update the general logfile
        with open(args.logfile_name,'a') as g:
            g.write(f'\tTrain loss at epoch {epoch_idx}, batch {batch_idx}: {batch_loss}\n')
        
        new_counter = 0
    
    return new_counter


###############################################################################
### Main function   ###########################################################
###############################################################################
def neural_train_loop( args,
                       epoch_arr,
                       all_trainstates,
                       training_rngkey,
                       have_acc,
                       dataloader_dict,
                       jitted_determine_seqlen_bin,
                       jitted_determine_alignlen_bin,
                       train_fn_jitted,
                       eval_fn_jitted,
                       all_save_model_filenames,
                       writer ): 
    ### model-specific prep BEFORE starting training loop
    # unpack dataloader
    training_dset = dataloader_dict['training_dset']
    training_dl = dataloader_dict['training_dl']
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    
    # init classes with memory, to help me keep track of things
    best_models_class = _best_models_neural( initial_trainstate_objs = all_trainstates )
    train_loop_timer_class = timers( num_epochs = args.num_epochs )
    eval_loop_timer_class = timers( num_epochs = args.num_epochs )
    whole_epoch_timer_class = timers( num_epochs = args.num_epochs )
    
    # track when jit compilations are made
    jit_compilation_in_train_tracking_class = jit_compilation_tracker( num_epochs = args.num_epochs )
    jit_compilation_in_eval_tracking_class = jit_compilation_tracker( num_epochs = args.num_epochs )
     
    
    ### start actual training loop
    for epoch_idx in tqdm(epoch_arr):    
        # classes to remember metrics THIS EPOCH
        # get re-initialized at the start of every epoch
        train_metrics_class = metrics_for_epoch( have_acc = have_acc,
                                                 epoch_idx = epoch_idx )  
        eval_metrics_class = metrics_for_epoch( have_acc = have_acc,
                                                 epoch_idx = epoch_idx ) 

        whole_epoch_timer_class.start_timer()

#__4___8: epoch level (two tabs)          
        ##############################################
        ### 3.1: train and update model parameters   #
        ##############################################
        train_loop_timer_class.start_timer()
        
        checkpoint_counter = -1
        for batch_idx, batch in enumerate(training_dl):
            this_batch_size = batch[0].shape[0]
            batch_epoch_idx = epoch_idx * len(training_dl) + batch_idx
            
#__4___8__12: batch level (three tabs) 
            # prep for training         
            out = _set_sequence_lengths_for_jit( args = args,
                                                 determine_seqlen_bin_fn = jitted_determine_seqlen_bin,
                                                 determine_alignlen_bin = jitted_determine_alignlen_bin,
                                                 batch = batch )
            jit_compilation_in_train_tracking_class.maybe_record_jit_compilation( clipped_lens = out,
                                                                         epoch_idx = epoch_idx )
            batch_max_seqlen, batch_max_alignlen = out
            del out
            
            rngkey_for_training_batch = jax.random.fold_in(training_rngkey, epoch_idx+batch_idx) 
            
            # run function to train on one batch of samples
            out = train_fn_jitted(batch=batch, 
                                  training_rngkey=rngkey_for_training_batch, 
                                  all_trainstates=all_trainstates, 
                                  max_seq_len = batch_max_seqlen,
                                  max_align_len = batch_max_alignlen)
            train_metrics, all_trainstates = out
            del out
            
            # if applicable: check for TKF approximation use
            if train_metrics.get('used_approx', None) is not None:
                subline = f'epoch {epoch_idx}, batch {batch_idx}:'
                write_approx_dict( approx_dict = train_metrics['used_approx'], 
                                   out_arrs_dir = args.out_arrs_dir,
                                   out_file = 'TRAIN_tkf_approx.tsv', 
                                   subline = subline,
                                   calc_sum = True )
            
            # potentially save the trainstates during training
            if args.checkpoint_freq_during_training is not None:
                checkpoint_counter = _maybe_checkpoint_neural_tstates_during_training( args = args,
                                                       prev_checkpoint_counter = checkpoint_counter,
                                                       epoch_idx = epoch_idx,
                                                       batch_idx = batch_idx,
                                                       batch_loss = train_metrics['batch_loss'],
                                                       all_trainstates = all_trainstates, 
                                                       all_save_model_filenames = all_save_model_filenames )
            
#__4___8__12: batch level (three tabs)
            ################################################################
            ### 3.2: if NaN is found, save current progress for inspection #
            ###      and quit training                                     #
            ################################################################
            if jnp.isnan( train_metrics['batch_loss'] ):
                # save the argparse object by itself
                args.epoch_idx = epoch_idx
                with open(f'{args.model_ckpts_dir}/TRAINING_ARGPARSE_BROKEN.pkl', 'wb') as g:
                    pickle.dump(args, g)
                
                # save the trainstates
                save_all_neural_trainstates( all_save_model_filenames = all_save_model_filenames,
                                      all_trainstates = all_trainstates, 
                                      suffix = 'BROKEN' )
                
                
                # one last recording to tensorboard
                write_optional_outputs_during_training(writer_obj = writer, 
                                                       all_trainstates = all_trainstates,
                                                       global_step = batch_epoch_idx, 
                                                       dict_of_values = train_metrics, 
                                                       interms_for_tboard = args.interms_for_tboard, 
                                                       write_histograms_flag = False)
                
                # record timing so far (if any)
                write_timing_file( outdir = args.logfile_dir,
                                   train_times = train_loop_timer_class.all_times,
                                   eval_times = eval_loop_timer_class.all_times,
                                   total_times = whole_epoch_timer_class.all_times,
                                   train_jit_epochs = jit_compilation_in_train_tracking_class.epochs_with_jit_comp,
                                   eval_jit_epochs = jit_compilation_in_eval_tracking_class.epochs_with_jit_comp )
                
                raise RuntimeError( ('NaN loss detected; saved intermediates '+
                                    'and quit training') )
                
            
#__4___8__12: batch level (three tabs)
            ### add to recorded metrics for this epoch
            train_metrics_class.update_after_batch( batch_weight = this_batch_size / len(training_dset),
                                        batch_loss = train_metrics['batch_loss'],
                                        batch_perpl = train_metrics['batch_ave_perpl'],
                                        batch_acc = train_metrics.get('batch_ave_acc', None) )
            
            write_optional_outputs_during_training(writer_obj = writer, 
                                                    all_trainstates = all_trainstates,
                                                    global_step = batch_epoch_idx, 
                                                    dict_of_values = train_metrics, 
                                                    interms_for_tboard = args.interms_for_tboard, 
                                                    write_histograms_flag = False)
            
#__4___8: epoch level (two tabs)
        ### manage timing
        train_loop_timer_class.end_timer_and_write_to_tboard( epoch_idx = epoch_idx,
                                                              writer = writer,
                                                              tag = 'Process training data' )
        
        
        ##############################################################
        ### 3.3: also check current performance on held-out test set #
        ##############################################################
        # start timer
        eval_loop_timer_class.start_timer()
        
        for batch_idx, batch in enumerate(test_dl):
            this_batch_size = batch[0].shape[0]
            
            # prep for eval         
            out = _set_sequence_lengths_for_jit( args = args,
                                                determine_seqlen_bin_fn = jitted_determine_seqlen_bin,
                                                determine_alignlen_bin = jitted_determine_alignlen_bin,
                                                batch = batch )
            jit_compilation_in_eval_tracking_class.maybe_record_jit_compilation( clipped_lens = out,
                                                                         epoch_idx = epoch_idx )
            batch_max_seqlen, batch_max_alignlen = out
            del out
            
            # run eval function
            eval_metrics = eval_fn_jitted(batch=batch, 
                                          all_trainstates=all_trainstates,
                                          max_seq_len=batch_max_seqlen,
                                          max_align_len=batch_max_alignlen)
            
#__4___8__12: batch level (three tabs)
            ### add to total loss for this epoch; weight by number of
            ###   samples/valid tokens in this batch
            eval_metrics_class.update_after_batch( batch_weight = this_batch_size / len(test_dset),
                                       batch_loss = eval_metrics['batch_loss'],
                                       batch_perpl = eval_metrics['batch_ave_perpl'],
                                       batch_acc = eval_metrics.get('batch_ave_acc', None) )
            
            
#__4___8: epoch level (two tabs) 
        ### manage timing
        eval_loop_timer_class.end_timer_and_write_to_tboard( epoch_idx = epoch_idx,
                                                             writer = writer,
                                                             tag = 'Process test data' )
        
        
        ##########################################
        ### 3.4: record scalars to tensorboard   #
        ##########################################
        train_metrics_class.write_epoch_metrics_to_tensorboard( writer = writer,
                                                  tag = 'training set')
        
        eval_metrics_class.write_epoch_metrics_to_tensorboard( writer = writer,
                                                 tag = 'test set')
        
        
        
#__4___8: epoch level (two tabs) 
        ##########################################################
        ### 3.5: if this is the best epoch TEST loss,            #
        ###      save the model params and args for later eval   #
        ##########################################################
        best_models_class.maybe_save_best_model( args = args,
                                            epoch_loss = eval_metrics_class.epoch_ave_loss,
                                            epoch_idx = epoch_idx,
                                            all_trainstates = all_trainstates,
                                            all_save_model_filenames = all_save_model_filenames )
        
        
#__4___8: epoch level (two tabs) 
        ###########################
        ### 3.6: EARLY STOPPING   #
        ###########################
        early_stop = best_models_class.check_early_stop( args = args,
                                                         epoch_loss = eval_metrics_class.epoch_ave_loss )
        
        if early_stop:
            # record in the raw ascii logfile
            with open(args.logfile_name,'a') as g:
                g.write(f'\n\nEARLY STOPPING AT {epoch_idx}:\n')
            
            # record time spent at this epoch
            whole_epoch_timer_class.end_timer_and_write_to_tboard( epoch_idx = epoch_idx,
                                                                  writer = writer,
                                                                  tag = 'Process one epoch' )
            break

#__4___8: epoch level (two tabs) 
        ### before next epoch, do this stuff
        # remember this epoch's loss for next iteration
        best_models_class.prev_test_loss = eval_metrics_class.epoch_ave_loss
        
        # record time spent at this epoch
        whole_epoch_timer_class.end_timer_and_write_to_tboard( epoch_idx = epoch_idx,
                                                              writer = writer,
                                                              tag = 'Process one epoch' )
    
    ### record timing at the end of all this
    write_timing_file( outdir = args.logfile_dir,
                       train_times = train_loop_timer_class.all_times,
                       eval_times = eval_loop_timer_class.all_times,
                       total_times = whole_epoch_timer_class.all_times,
                       train_jit_epochs = jit_compilation_in_train_tracking_class.epochs_with_jit_comp,
                       eval_jit_epochs = jit_compilation_in_eval_tracking_class.epochs_with_jit_comp )
    
    return (early_stop, best_models_class)
