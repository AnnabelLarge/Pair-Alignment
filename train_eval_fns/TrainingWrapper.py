#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 17:42:19 2025

@author: annabel
"""
import jax
from jax import numpy as jnp
import flax
import numpy as np
import pickle
import sys
from tqdm import tqdm
from functools import partial
from copy import copy
from datetime import datetime

from utils.train_eval_utils import ( timers,
                                     write_timing_file,
                                     metrics_for_epoch,
                                     write_approx_dict,
                                     jit_compile_determine_seqlen_bin,
                                     jit_compile_determine_alignlen_bin,
                                     jit_compilation_tracker )

from utils.tensorboard_recording_utils import (write_optional_outputs_during_training,
                                               write_optional_outputs_during_training_hmms)


###############################################################################
### Base class  ###############################################################
###############################################################################
class TrainingWrapper:
    def __init__(self,
                 args,
                 epoch_arr,
                 initial_training_rngkey,
                 dataloader_dict,
                 train_fn_jitted,
                 eval_fn_jitted,
                 all_save_model_filenames,
                 writer): 
        ### read arguments
        # initialize as-is
        self.args = args
        self.epoch_arr = epoch_arr
        self.rngkey = initial_training_rngkey
        self.train_fn_jitted = train_fn_jitted
        self.eval_fn_jitted = eval_fn_jitted
        self.all_save_model_filenames = all_save_model_filenames
        self.writer = writer
        self.use_scan_fns = getattr(args, "use_scan_fns", False)
        
        # unpack dataloader dict
        self.training_dset = dataloader_dict['training_dset']
        self.training_dl = dataloader_dict['training_dl']
        
        
        ### smaller classes with memory
        # time, checkpointing
        self.train_loop_timer = timers( num_epochs = len(self.epoch_arr) )
        self.eval_loop_timer = timers( num_epochs = len(self.epoch_arr) )
        self.whole_epoch_timer = timers( num_epochs = len(self.epoch_arr) )
        self.intermediate_training_checkpoint_counter = 1
        self.early_stopping_counter = 0
        
        # track when jit compilations are made
        self.train_jit_compilation_tracker = jit_compilation_tracker( num_epochs = len(self.epoch_arr) )
        self.eval_jit_compilation_tracker = jit_compilation_tracker( num_epochs = len(self.epoch_arr) )
        
        
        ### model-specific initializations
        self._model_specific_inits(dataloader_dict = dataloader_dict)


    ###########################################################################
    ### Main training function: called in CLIs   ##############################
    ###########################################################################
    def run_train_loop( self, 
                        all_trainstates ):
        # keep track of best model; need to know initial trainstate object
        best_epoch = -1
        best_trainstates = copy(all_trainstates)
        prev_dev_loss = jnp.finfo(jnp.float32).max
        best_dev_loss = jnp.finfo(jnp.float32).max
        
        # record loss trajectory to a flat text file too
        loss_file = f'{self.args.logfile_dir}/losses_flat.tsv'
        with open(loss_file,'w') as g:
            g.write( ('time\t' + 
                      'epoch\t' + 
                      'ave_train_loss\t' + 
                      'ave_dev_loss\t' + 
                      'best_model\n') )
        
        for epoch_idx in tqdm(self.epoch_arr): 
            ### train and update gradients
            out = self.train_and_eval_one_epoch( all_trainstates, epoch_idx )
            all_trainstates = out['tstates']
            epoch_train_loss = out['train_loss']
            epoch_dev_loss = out['dev_loss']
            del out
            
            # update loss text file
            with open(loss_file,'a') as g:
                best_so_far = epoch_dev_loss < best_dev_loss
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                g.write( (f'{now}\t'+
                          f'{epoch_idx}\t'+
                          f'{epoch_train_loss}\t'+
                          f'{epoch_dev_loss}\t' +
                          f'{best_so_far}\n') )
                del now
                
            
            ### if this is the best model, save it
            if best_so_far:
                with open(self.args.logfile_name,'a') as g:
                    g.write( f'New best dev loss at epoch {epoch_idx}: {epoch_dev_loss}\n' )
                
                # update "best" recordings
                best_dev_loss = epoch_dev_loss
                best_trainstates = copy(all_trainstates)
                best_epoch = epoch_idx
                
                # save the trainstates
                self._save_model( filenames = self.all_save_model_filenames,
                                  trainstate_objs = best_trainstates,
                                  suffix = 'BEST' )
            
            
            ### check if early stopping conditions are met
            early_stop = self._maybe_early_stop(prev_loss = prev_dev_loss,
                                                curr_loss = epoch_dev_loss,
                                                best_loss = best_dev_loss)
            
            if early_stop:
                # record in the raw ascii logfile
                with open(self.args.logfile_name,'a') as g:
                    g.write(f'\n\nEARLY STOPPING AT {epoch_idx}:\n')
                
                break

            # remember this epoch's loss for next iteration
            prev_dev_loss = epoch_dev_loss
        
        
        ### record timing at the end of all this
        write_timing_file( outdir = self.args.logfile_dir,
                           train_times = self.train_loop_timer.all_times,
                           eval_times = self.eval_loop_timer.all_times,
                           total_times = self.whole_epoch_timer.all_times,
                           train_jit_epochs = self.train_jit_compilation_tracker.epochs_with_jit_comp,
                           eval_jit_epochs = self.eval_jit_compilation_tracker.epochs_with_jit_comp )
        
        return (early_stop, best_epoch, best_trainstates)


    ###########################################################################
    ### Epoch-level   #########################################################
    ###########################################################################
    def train_and_eval_one_epoch( self, 
                                  all_trainstates,
                                  epoch_idx ):
        ### metrics for the WHOLE DATASET; these get re-initialized at the 
        ###   start of every epoch
        train_metrics_recorder = metrics_for_epoch( have_acc = self.have_acc,
                                                 epoch_idx = epoch_idx )  
        dev_metrics_recorder = metrics_for_epoch( have_acc = self.have_acc,
                                                 epoch_idx = epoch_idx ) 
        self.whole_epoch_timer.start_timer()
        
        
        ### train set: evaluate loss and update gradients
        for batch_idx, batch in enumerate(self.training_dl):
            out = self.train_one_batch( previous_trainstates = all_trainstates,
                                        batch = batch,
                                        epoch_idx = epoch_idx,
                                        batch_idx = batch_idx )
            train_metrics, all_trainstates, train_batch_size = out
            del out
            
            self.save_train_metrics( metrics = train_metrics,
                                     metrics_recorder = train_metrics_recorder,
                                     updated_trainstates = all_trainstates,
                                     this_batch_size = train_batch_size,
                                     epoch_idx = epoch_idx,
                                     batch_idx = batch_idx )
            del train_batch_size, train_metrics
            
        
        ### dev set: just evaluate
        for batch_idx, batch in enumerate(self.dev_dl):
            out = self.eval_one_batch( all_trainstates = all_trainstates,
                                       batch = batch,
                                       epoch_idx = epoch_idx,
                                       batch_idx = batch_idx )
            eval_metrics, eval_batch_size = out
            del out
            
            self.save_eval_metrics( metrics = eval_metrics,
                                     metrics_recorder = dev_metrics_recorder,
                                     all_trainstates = all_trainstates,
                                     this_batch_size = eval_batch_size,
                                     epoch_idx = epoch_idx,
                                     batch_idx = batch_idx )
            del eval_metrics, eval_batch_size
        
        
        ### final records for the epoch
        # record metrics to tensorboard
        train_metrics_recorder.write_epoch_metrics_to_tensorboard( writer = self.writer,
                                                                   tag = 'training set')
        
        dev_metrics_recorder.write_epoch_metrics_to_tensorboard( writer = self.writer,
                                                                  tag = 'dev set')
            
        # record time spent at this epoch
        self.whole_epoch_timer.end_timer_and_write_to_tboard( epoch_idx = epoch_idx,
                                                              writer = self.writer,
                                                              tag = 'Process one epoch' )
        
        return {'tstates': all_trainstates,
                'train_loss': train_metrics_recorder.epoch_ave_loss, 
                'dev_loss': dev_metrics_recorder.epoch_ave_loss}
    

    ###########################################################################
    ### Batch-level   #########################################################
    ###########################################################################
    def train_one_batch( self, 
                         previous_trainstates,
                         batch,
                         epoch_idx,
                         batch_idx ):
        this_batch_size = batch[0].shape[0]

        # handle random key
        self.rngkey, rngkey_for_batch = jax.random.split(self.rngkey)
        batch_epoch_idx = epoch_idx * len(self.training_dl) + batch_idx
        rngkey_for_batch = jax.random.fold_in(rngkey_for_batch, batch_epoch_idx) 
        
        # prep sequence lengths; this determines whether or not you jit-compile 
        out = self._set_sequence_lengths_for_jit( batch = batch )
        batch_max_seqlen, batch_max_alignlen = out
        
        if batch_max_seqlen is not None:
            batch_max_seqlen = batch_max_seqlen.item()
        if batch_max_alignlen is not None:
            batch_max_alignlen = batch_max_alignlen.item()
            
        size_tuple = (this_batch_size, batch_max_seqlen, batch_max_alignlen)
        self.train_jit_compilation_tracker.maybe_record_jit_compilation( size_tuple = size_tuple,
                                                                         epoch_idx = epoch_idx )
        del out, size_tuple
        
        # start a timer
        self.train_loop_timer.start_timer()
        
        # run function to train on one batch of samples
        out = self.train_fn_jitted(batch=batch, 
                              training_rngkey = rngkey_for_batch, 
                              all_trainstates = previous_trainstates, 
                              max_seq_len = batch_max_seqlen,
                              max_align_len = batch_max_alignlen)
        train_metrics, updated_trainstates = out
        del out
        
        # end timer
        self.train_loop_timer.end_timer_and_write_to_tboard( epoch_idx = epoch_idx,
                                                             writer = self.writer,
                                                             tag = 'Process training data' )
        
        # check for nan loss; will quit if this happens
        self._check_for_nan_train_loss( loss = train_metrics['batch_loss'],
                                        epoch_idx = epoch_idx,
                                        trainstate_objs = updated_trainstates )
        
        return train_metrics, updated_trainstates, this_batch_size
    
    def save_train_metrics( self, 
                            metrics,
                            metrics_recorder,
                            updated_trainstates,
                            this_batch_size,
                            epoch_idx,
                            batch_idx ):
        batch_epoch_idx = epoch_idx * len(self.training_dl) + batch_idx
        
        # if applicable: check for TKF approximation used
        if (self.use_tkf_funcs) and (metrics['used_approx'] is not None):
            subline = f'epoch {epoch_idx}, batch {batch_idx}:'
            self.write_approx_dict_fn( approx_dict = metrics['used_approx'], 
                                       subline = subline )
            del subline
        
        # potentially save the trainstates during training; update the 
        #   checkpoint counter
        self._maybe_checkpoint( epoch_idx = epoch_idx,
                                batch_idx = batch_idx,
                                batch_loss = metrics['batch_loss'],
                                trainstate_objs = updated_trainstates )
        
        # record metrics
        metrics_recorder.update_after_batch( batch_weight = this_batch_size / len(self.training_dset),
                                    batch_loss = metrics['batch_loss'],
                                    batch_acc = metrics.get('batch_ave_acc', None) )
        
        self.optional_outputs_writer(writer_obj = self.writer, 
                                                all_trainstates = updated_trainstates,
                                                global_step = batch_epoch_idx, 
                                                dict_of_values = metrics, 
                                                interms_for_tboard = self.args.interms_for_tboard, 
                                                write_histograms_flag = False)

    def eval_one_batch( self, 
                         all_trainstates,
                         batch,
                         epoch_idx,
                         batch_idx ):
        this_batch_size = batch[0].shape[0]

        # prep sequence lengths; this determines whether or not you jit-compile 
        #   again
        out = self._set_sequence_lengths_for_jit( batch = batch )
        batch_max_seqlen, batch_max_alignlen = out
        
        if batch_max_seqlen is not None:
            batch_max_seqlen = batch_max_seqlen.item()
        if batch_max_alignlen is not None:
            batch_max_alignlen = batch_max_alignlen.item()
        
        size_tuple = (this_batch_size, batch_max_seqlen, batch_max_alignlen)
        self.eval_jit_compilation_tracker.maybe_record_jit_compilation( size_tuple = size_tuple,
                                                                         epoch_idx = epoch_idx )
        del out, size_tuple
        
        # start a timer; will end it after recording metrics
        self.eval_loop_timer.start_timer()
        
        # run function to evaluate on one batch of samples
        eval_metrics = self.eval_fn_jitted(batch=batch, 
                                           all_trainstates = all_trainstates, 
                                           max_seq_len = batch_max_seqlen,
                                           max_align_len = batch_max_alignlen)
        
        # end timer
        self.eval_loop_timer.end_timer_and_write_to_tboard( epoch_idx = epoch_idx,
                                                             writer = self.writer,
                                                             tag = 'Process dev data' )
        return (eval_metrics, this_batch_size)
    
    def save_eval_metrics( self, 
                            metrics,
                            metrics_recorder,
                            all_trainstates,
                            this_batch_size,
                            epoch_idx,
                            batch_idx):
        batch_epoch_idx = epoch_idx * len(self.dev_dl) + batch_idx
        
        metrics_recorder.update_after_batch( batch_weight = this_batch_size / len(self.dev_dset),
                                    batch_loss = metrics['batch_loss'],
                                    batch_acc = metrics.get('batch_ave_acc', None) )
        
        no_returns = {k: False for k in self.args.interms_for_tboard.keys()}
        
        self.optional_outputs_writer(writer_obj = self.writer, 
                                                all_trainstates = all_trainstates,
                                                global_step = batch_epoch_idx, 
                                                dict_of_values = metrics, 
                                                interms_for_tboard = no_returns, 
                                                write_histograms_flag = False)
    

    ###########################################################################
    ### Internal helpers   ####################################################
    ###########################################################################
    def _save_model( self,
                     filenames: list,
                     trainstate_objs: list,
                     suffix = None ):
        for i in range(len(trainstate_objs)):
            new_outfile = filenames[i]
            
            if suffix is not None:
                new_outfile = new_outfile.replace('.pkl',f'_{suffix}.pkl')
            
            with open(new_outfile, 'wb') as g:
                model_state_dict = flax.serialization.to_state_dict(trainstate_objs[i])
                pickle.dump(model_state_dict, g)    
    
    def _maybe_checkpoint( self,
                           epoch_idx,
                           batch_idx,
                           batch_loss,
                           trainstate_objs ):
        if self.args.checkpoint_freq_during_training > 0:
            flag = ( self.intermediate_training_checkpoint_counter % self.args.checkpoint_freq_during_training == 0 )
        else:
            flag = False
        
        if flag:
            # save some metadata about the trainstate files
            with open(f'{self.args.model_ckpts_dir}/INPROGRESS_trainstates_info.txt','w') as g:
                g.write(f'Checkpoint created at: epoch {epoch_idx}, batch {batch_idx}\n')
                g.write(f'Current loss for the training set batch is: {batch_loss}\n')
            
            # save the trainstates
            self._save_model( filenames = self.all_save_model_filenames,
                              trainstate_objs = trainstate_objs, 
                              suffix = 'INPROGRESS' )
            
            # update the general logfile
            with open(self.args.logfile_name,'a') as g:
                g.write(f'\tTrain loss at epoch {epoch_idx}, batch {batch_idx}: {batch_loss}\n')
            
            self.intermediate_training_checkpoint_counter = 1
        
        else:
            self.intermediate_training_checkpoint_counter += 1
            
    def _maybe_early_stop(self,
                          prev_loss,
                          curr_loss,
                          best_loss):
        # condition 1: if dev loss stagnates or starts to go up, compared
        #              to previous epoch's dev loss
        cond1 = jnp.allclose( prev_loss, 
                              jnp.minimum (prev_loss, curr_loss), 
                              atol=self.args.early_stop_cond1_atol,
                              rtol=0 )

        # condition 2: if dev loss is substatially worse than best dev loss
        cond2 = (curr_loss - best_loss) > self.args.early_stop_cond2_gap

        if cond1 or cond2:
            self.early_stopping_counter += 1
        else:
            self.early_stopping_counter = 0
        
        return (self.early_stopping_counter  == self.args.patience)
    
    def _check_for_nan_train_loss( self, 
                                   loss,
                                   epoch_idx,
                                   trainstate_objs ):
        if jnp.isnan( loss ):
            # save the argparse object by itself
            self.args.epoch_idx = epoch_idx
            with open(f'{self.args.model_ckpts_dir}/TRAINING_ARGPARSE_BROKEN.pkl', 'wb') as g:
                pickle.dump(self.args, g)
            
            # save the trainstates
            self._save_model( filenames = self.all_save_model_filenames,
                              trainstate_objs = trainstate_objs,
                              suffix = 'BROKEN' )
            
            # record timing so far (if any)
            write_timing_file( outdir = self.args.logfile_dir,
                               train_times = self.train_loop_timer.all_times,
                               eval_times = self.eval_loop_timer.all_times,
                               total_times = self.whole_epoch_timer.all_times,
                               train_jit_epochs = self.train_jit_compilation_tracker.epochs_with_jit_comp,
                               eval_jit_epochs = self.eval_jit_compilation_tracker.epochs_with_jit_comp )
            
            raise RuntimeError( ('NaN loss detected; saved intermediates '+
                                'and quit training') )
    
    def _set_sequence_lengths_for_jit( self, 
                                       batch ):
        # unpack briefly to get max len and number of samples in the 
        #   batch; place in some bin (this controls how many jit 
        #   compilations you do)
        batch_max_seqlen = self.seqlen_bin_fn(batch = batch)
        batch_max_alignlen = self.alignlen_bin_fn(batch = batch)
        
        # I've had so much trouble with this ugh
        if self.use_scan_fns:
            err = (f'batch_max_alignlen (not including bos) is: '+
                   f'{batch_max_alignlen - 1}'+
                   f', which is not divisible by length for scan '+
                   f'({self.args.chunk_length})')
            assert (batch_max_alignlen - 1) % self.args.chunk_length == 0, err
        
        return batch_max_seqlen, batch_max_alignlen
    
    def _model_specific_inits(self):
        raise NotImplementedError('depends on model!')



###############################################################################
### Model-specific subclasses  ################################################
###############################################################################
class NeuralTKFTrainingWrapper(TrainingWrapper):
    def _model_specific_inits(self, dataloader_dict):
        # check model type again
        assert self.args.pred_model_type == 'neural_hmm'
        
        # add dev set
        self.dev_dset = dataloader_dict['dev_dset']
        self.dev_dl = dataloader_dict['dev_dl']
        
        # continue init
        self.seqlen_bin_fn = jit_compile_determine_seqlen_bin(self.args)
        self.alignlen_bin_fn = jit_compile_determine_alignlen_bin(self.args)
        self.have_acc = False
        self.use_tkf_funcs = True
        self.write_approx_dict_fn = partial( write_approx_dict,
                                            calc_sum = True,
                                            out_arrs_dir = self.args.out_arrs_dir,
                                            out_file = 'TRAIN_tkf_approx.tsv' )
        self.optional_outputs_writer = write_optional_outputs_during_training

class FeedforwardTrainingWrapper(TrainingWrapper):
    def _model_specific_inits(self, dataloader_dict):
        # check model type again
        assert self.args.pred_model_type == 'feedforward'
        
        # add dev set
        self.dev_dset = dataloader_dict['dev_dset']
        self.dev_dl = dataloader_dict['dev_dl']
        
        # continue init
        self.seqlen_bin_fn = jit_compile_determine_seqlen_bin(self.args)
        self.alignlen_bin_fn = jit_compile_determine_alignlen_bin(self.args)
        self.have_acc = True
        self.use_tkf_funcs = False
        self.write_approx_dict_fn = lambda *args, **kwargs: None
        self.optional_outputs_writer = write_optional_outputs_during_training


class TransitMixesTrainingWrapper(TrainingWrapper):
    def _model_specific_inits(self, dataloader_dict):
        # check model type again
        assert self.args.pred_model_type in ['pairhmm_frag_and_site_classes',
                                             'pairhmm_nested_tkf']
        
        # replace "dev set" with a copy of the test set
        self.dev_dset = dataloader_dict['test_dset']
        self.dev_dl = dataloader_dict['test_dl']
        
        # continue init
        self.seqlen_bin_fn = lambda *args, **kwargs: None
        self.alignlen_bin_fn = jit_compile_determine_alignlen_bin(self.args)
        self.have_acc = False
        self.use_tkf_funcs = True
        self.write_approx_dict_fn = partial( write_approx_dict,
                                            calc_sum = False,
                                            out_arrs_dir = self.args.out_arrs_dir,
                                            out_file = 'TRAIN_tkf_approx.tsv' )
        self.optional_outputs_writer = write_optional_outputs_during_training_hmms

class IndpSitesTrainingWrapper(TrainingWrapper):
    def _model_specific_inits(self, dataloader_dict):
        # check model type again
        assert self.args.pred_model_type == 'pairhmm_indp_sites'
        
        # replace "dev set" with a copy of the test set
        self.dev_dset = dataloader_dict['test_dset']
        self.dev_dl = dataloader_dict['test_dl']
        
        # continue init
        self.seqlen_bin_fn = lambda *args, **kwargs: None
        self.alignlen_bin_fn = lambda *args, **kwargs: None
        self.have_acc = False
        self.use_tkf_funcs = True
        self.write_approx_dict_fn = partial( write_approx_dict,
                                            calc_sum = False,
                                            out_arrs_dir = self.args.out_arrs_dir,
                                            out_file = 'TRAIN_tkf_approx.tsv' )
        self.optional_outputs_writer = write_optional_outputs_during_training_hmms
