#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:12:41 2025

@author: annabel

ABOUT: 
======
main training loop

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
from utils.edit_argparse import (enforce_valid_defaults,
                                 fill_with_default_values,
                                 share_top_level_args)
from utils.setup_training_dir import setup_training_dir
from train_eval_fns.build_optimizer import build_optimizer
from utils.sequence_length_helpers import (determine_seqlen_bin, 
                                           determine_alignlen_bin)
from utils.tensorboard_recording_utils import (write_times,
                                               write_optional_outputs_during_training)



def train(args, dataloader_dict: dict):
    ###########################################################################
    ### 0: CHECK CONFIG; IMPORT APPROPRIATE MODULES   #########################
    ###########################################################################
    ### edit the argparse object in-place
    enforce_valid_defaults(args)
    fill_with_default_values(args)
    share_top_level_args(args)
    
    
    ### import train/test functions; enforce some default variables
    if args.pred_model_type == 'feedforward':    
        have_acc = True
        args.times_from = None
        
        if args.use_scan_fns:
            raise NotImplementedError('Add loss function')
        elif not args.use_scan_fns:
            raise NotImplementedError('Add loss function')
    
    elif args.pred_model_type == 'neural_pairhmm':  
        have_acc = False
        if args.use_scan_fns:
            raise NotImplementedError('Add loss function')
        elif not args.use_scan_fns:
            raise NotImplementedError('Add loss function')
    
    elif args.pred_model_type == 'pairhmm':
        have_acc = False
        raise NotImplementedError('Add loss function')
    
    
    ### final defaults
    # marginalizing over time?
    if args.pred_model_type in ['neural_pairhmm', 'pairhmm']:
        have_time_array = True
    elif args.pred_model_type in ['feedforward']:
        have_time_array = False
    
    # using full alignment lengths, or summary of counts?
    if args.pred_model_type in ['feedforward', 'neural_pairhmm']:
        have_full_length_alignments = True
    elif args.pred_model_type in ['pairhmm']:
        have_full_length_alignments = False
    
    
    
    ###########################################################################
    ### 1: SETUP   ############################################################
    ###########################################################################
    print(f'1: setup')
    with open(args.logfile_name,'a') as g:
        g.write(f'1: setup\n')
        
    ### initial setup of misc things
    # setup the working directory (if not done yet) and this run's sub-directory
    setup_training_dir(args)
    
    # initial random key, to carry through execution
    rngkey = jax.random.key(args.rng_seednum)
    
    # setup tensorboard writer
    writer = SummaryWriter(args.tboard_dir)
    
    # create a new logfile
    with open(args.logfile_name,'w') as g:
        g.write(f'Ancestor Embedder: {args.anc_model_type}\n')
        g.write(f'Descendant Embedder: {args.desc_model_type}\n')
        g.write(f'Likelihood model: {args.pred_model_type}\n')
        g.write(f'Normalizing losses by: {args.norm_loss_by}\n')
    
    ### !!!!!!!!!!!!!!! BELOW IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
    # create somewhere to write losses as you train (because tensorboard)
    #   might not display everything....
    with open(f'{args.logfile_dir}/LOSSES.tsv','w') as g:
        g.write( ('epoch_idx' + '\t' +
                  'which' + '\t' +
                  'batch_idx' + '\t' +
                  'ave_loss' + '\t' +
                  'ave_perplexity' + '\t' +
                  'ece' + '\t' +
                  'notes' + '\n')
                )
    ### !!!!!!!!!!!!!!! ABOVE IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
        
        
    ### extract data from dataloader_dict
    training_dset = dataloader_dict['training_dset']
    training_dl = dataloader_dict['training_dl']
    test_dset = dataloader_dict['test_dset']
    test_dl = dataloader_dict['test_dl']
    
    # add the equilibrium distribution from the training dataset object
    #   to the argparse object
    if args.pred_model_type == 'neural_pairhmm':
        args.pred_config['equilibr_config']['training_dset_aa_counts'] = training_dset.aa_counts
    del dataloader_dict
    
    
    ### setup for saving model parameters with pickle
    # counts-based models will save empty pickles
    encoder_save_model_filename = args.model_ckpts_dir + '/'+ f'ANC_ENC.pkl'
    decoder_save_model_filename = args.model_ckpts_dir + '/'+ f'DESC_DEC.pkl'
    finalpred_save_model_filename = args.model_ckpts_dir + '/'+ f'FINAL_PRED.pkl'
    all_save_model_filenames = [encoder_save_model_filename, 
                                decoder_save_model_filename,
                                finalpred_save_model_filename]
    
    
    ### write argparse when done
    write_config(args = args, 
                 out_dir = args.model_ckpts_dir)
    
    
    ###########################################################################
    ### 2: INITIALIZE MODEL PARTS, OPTIMIZER  #################################
    ###########################################################################
    print('2: model init')
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'2: model init\n')
    
    # init the optimizer
    tx = build_optimizer(args)
    
    # initialize dummy array of times
    if have_time_array:
        num_timepoints = training_dset.retrieve_num_timepoints(times_from = args.times_from)
        dummy_t_array = jnp.empty( (num_timepoints, args.batch_size) )
    
    elif not have_time_array:
        dummy_t_array = None
    
    # split a new rng key
    rngkey, model_init_rngkey = jax.random.split(rngkey, num=2)
    
    
    ### for models that use full-length alignments: determine lengths
    if have_full_length_alignments:
        # for unaligned seqs: longest possible input is global_seq_max_length
        global_seq_max_length = max([training_dset.global_seq_max_length,
                                     test_dset.global_seq_max_length])
        largest_seqs = (args.batch_size, global_seq_max_length)
        
        # for aligned matrices: longest possible input depends on if you're
        #   using the scanned version of likelihood fns
        if args.use_scan_fns:
            max_dim1 = args.chunk_length
        
        elif not args.use_scan_fns:
            max_dim1 = max([training_dset.global_align_max_length,
                            test_dset.global_align_max_length]) - 1
          
        largest_aligns = (args.batch_size, max_dim1)
        del max_dim1
        
        seq_shapes = [largest_seqs, largest_aligns]
        
        # parted and jit-compiled seq len helpers
        parted_determine_seqlen_bin = partial(determine_seqlen_bin,
                                              chunk_length = args.chunk_length, 
                                              seq_padding_idx = args.seq_padding_idx)
        jitted_determine_seqlen_bin = jax.jit(parted_determine_seqlen_bin)
        del parted_determine_seqlen_bin
        
        parted_determine_alignlen_bin = partial(determine_alignlen_bin,  
                                                chunk_length = args.chunk_length,
                                                seq_padding_idx = args.seq_padding_idx)
        jitted_determine_alignlen_bin = jax.jit(parted_determine_alignlen_bin)
        del parted_determine_alignlen_bin
    
    
    ### for models that use full-length alignments: define input shapes
    elif not have_full_length_alignments:
        B = args.batch_size
        emission_alphabet = args.base_alphabet_size - 3
        num_states = 3 if args.bos_eos_as_match else 5
        
        dummy_subCounts = jnp.empty( (B, emission_alphabet, emission_alphabet) )
        dummy_insCounts = jnp.empty( (B, emission_alphabet) )
        dummy_delCounts = jnp.empty( (B, emission_alphabet) )
        dummy_transCounts = jnp.empty( (B, num_states, num_states) )
        
        seq_shapes = [dummy_subCounts,
                      dummy_insCounts,
                      dummy_delCounts,
                      dummy_transCounts]
        
        jitted_determine_seqlen_bin = None
        jitted_determine_alignlen_bin = None
        
    
    raise NotImplementedError('Make initializers for each model AFTER defining models')
    """
    version of function in neural codebase:
    
    out = create_all_tstates(seq_shapes = seq_shapes, 
                              dummy_t_array = dummy_t_array,
                              tx = tx, 
                              model_init_rngkey = model_init_rngkey,
                              tabulate_file_loc = args.model_ckpts_dir,
              [remove this ] which_alignment_states_to_encode = args.which_alignment_states_to_encode,
                              anc_model_type = args.anc_model_type, 
                              desc_model_type = args.desc_model_type, 
                              pred_model_type = args.pred_model_type, 
                              anc_enc_config = args.anc_enc_config, 
                              desc_dec_config = args.desc_dec_config, 
                              pred_config = args.pred_config,
                              )  
    
    
    pairHMM will just use a predictive head that: 
      1.) creates global exch mat (or read LG08 exch)
      2.) creates global equilib dist (or calculates from counts)
      3.) initializes appropriate indel parameters per indel model
    
    and have no sequence embedders
    """
    
    
    ###  part and jit the functions for the training loop
    """
    taken from neural codebase:
    NEED TO ADD: args.loss_type (which is either conditional or joint), 
                 args.num_site_classes (just one, for now, but need the option to expand later)
    
    place everything below in a separate wrapper call
    def initialize_train_fn( train_function,
                             argparse_object, 
                             all_model_instances ):
        ...
        return train_fn_jitted
    
    def initialize_eval_fn( eval_function,
                            argparse_object, 
                            all_model_instances,
                            extra_args_for_eval,
                            for_training: bool ):
        ...
        return eval_fn_jitted
    
    ### training_fn
    parted_train_fn = partial( train_one_batch,
                               all_model_instances = all_model_instances,
                               which_alignment_states_to_encode = args.which_alignment_states_to_encode,
                               interms_for_tboard = args.interms_for_tboard,
                               norm_loss_by = args.pred_config['norm_loss_by'],
                               length_for_scan = args.chunk_length,
                               have_time_values = have_time_values,
                               seq_padding_idx = args.seq_padding_idx
                              )
    
    train_fn_jitted = jax.jit(parted_train_fn, 
                              static_argnames = ['max_seq_len',
                                                  'max_align_len'])
    del parted_train_fn
    
    
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
                  'forward_pass_outputs': False,
                  'final_logprobs': False}
    extra_args_for_eval = dict()
    
    # if this is a transformer model, will have extra arguments for eval funciton
    if (args.anc_model_type == 'Transformer' or args.desc_model_type == 'Transformer'):
        extra_args_for_eval['output_attn_weights'] = False
    
    parted_eval_fn = partial(eval_one_batch,
                             all_model_instances = all_model_instances,
                             which_alignment_states_to_encode = args.which_alignment_states_to_encode, 
                             interms_for_tboard = no_returns,
                             length_for_scan = args.chunk_length,
                             seq_padding_idx = args.seq_padding_idx,
                             norm_loss_by = args.pred_config['norm_loss_by'],
                             have_time_values = have_time_values,
                             extra_args_for_eval = extra_args_for_eval)
    del no_returns, extra_args_for_eval
    
    # jit compile this eval function
    eval_fn_jitted = jax.jit(parted_eval_fn, 
                              static_argnames = ['max_seq_len',
                                                 'max_align_len'])
    del parted_eval_fn
    """
    
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
    best_trainstates = tuple()
    if have_acc:
        best_test_acc = -1
    
    # quit training if test loss increases for X epochs in a row
    prev_test_loss = 999999
    early_stopping_counter = 0
    
    # rng key for train
    rngkey, training_rngkey = jax.random.split(rngkey, num=2)
    
    for epoch_idx in tqdm(range(args.num_epochs)):
        ### !!!!!!!!!!!!!!! BELOW IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
        # every so often, checkpoint the argparse object
        if epoch_idx == 5:
            args.epoch_idx = epoch_idx
            with open(f'{args.model_ckpts_dir}/TRAINING_ARGPARSE.pkl', 'wb') as g:
                pickle.dump(args, g)
        # !!!!!!!!!!!!!!! ABOVE IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 

        epoch_real_start = wall_clock_time()
        epoch_cpu_start = process_time()
        
        ave_epoch_train_loss = 0
        ave_epoch_train_perpl = 0
        if have_acc:
            ave_epoch_train_acc = 0

#__4___8: epoch level (two tabs)          
        ##############################################
        ### 3.1: train and update model parameters   #
        ##############################################
        for batch_idx, batch in enumerate(training_dl):
            batch_epoch_idx = epoch_idx * len(training_dl) + batch_idx
            batch_real_start = wall_clock_time()
            batch_cpu_start = process_time()

#__4___8__12: batch level (three tabs)          
            # unpack briefly to get max len and number of samples in the 
            #   batch; place in some bin (this controls how many jit 
            #   compilations you do)
            if have_full_length_alignments:
                batch_max_seqlen = jitted_determine_seqlen_bin(batch = batch)
                batch_max_seqlen = batch_max_seqlen.item()
                batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch)
                batch_max_alignlen = batch_max_alignlen.item()
                
                # I've had so much trouble with this ugh
                if args.use_scan_fns:
                    err = (f'batch_max_alignlen (not including bos) is: '+
                           f'{batch_max_alignlen - 1}'+
                           f', which is not divisible by length for scan '+
                           f'({args.chunk_length})')
                    assert (batch_max_alignlen - 1) % args.chunk_length == 0, err
            
            elif not have_full_length_alignments:
                batch_max_seqlen = None
                batch_max_alignlen = None
            
            # !!!!!!!!!!!!!!! BELOW IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
            # record values before gradient update
            if batch_epoch_idx > 0:
                for key, val in train_metrics.items():
                    if key.startswith('FPO_'):
                        out_filename = f'{args.out_arrs_dir}/{key.replace("FPO_","BEFORE-UPDATE_")}.npy'
                        with open(out_filename,'wb') as g:
                            np.save(g, val)
            # !!!!!!!!!!!!!!! ABOVE IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
            
            rngkey_for_training_batch = jax.random.fold_in(training_rngkey, epoch_idx+batch_idx)
            
            
            ### run function to train on one batch of samples
            raise NotImplementedError('make and jit-compile a training function')
            out = train_fn_jitted(batch=batch, 
                                  training_rngkey=rngkey_for_training_batch, 
                                  all_trainstates=all_trainstates, 
                                  max_seq_len = batch_max_seqlen,
                                  max_align_len = batch_max_alignlen )
            train_metrics, all_trainstates = out
            del out
            
            
            # !!!!!!!!!!!!!!! BELOW IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
            ### DEBUG: log loss per batch in a text file
            batch_samples = training_dset.retrieve_sample_names(batch[-1])
            with open(f'{args.logfile_dir}/LOSSES.tsv','a') as g:
                g.write( (f'{epoch_idx}' + '\t' +
                          f'train_set' + '\t' +
                          f'{batch_idx}' + '\t' +
                          f"{train_metrics['batch_loss']}" + '\t' +
                          f"{train_metrics['batch_ave_perpl']}" + '\t' +
                          f"{np.exp(train_metrics['batch_loss'])}" + '\t' +
                          f'none' + '\n' )
                        )
            
            ### DEBUG: record output after gradient update
            # save any forward-pass outputs
            for key, val in train_metrics.items():
                if key.startswith('FPO_'):
                    out_filename = (f'{args.out_arrs_dir}/'+
                                    '{key.replace("FPO_","AFTER-UPDATE_")}.npy')
                    with open(out_filename,'wb') as g:
                        np.save(g, val)
            # !!!!!!!!!!!!!!! ABOVE IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
            
            
#__4___8__12: batch level (three tabs)
            ################################################################
            ### 3.2: if NaN is found, save current progress for inspection #
            ###      and quit training                                     #
            ################################################################
            if jnp.isnan( train_metrics['batch_loss'] ):
                # save the argparse object by itself
                args.epoch_idx = epoch_idx
                with open(f'{args.model_ckpts_dir}/TRAINING_ARGPARSE.pkl', 'wb') as g:
                    pickle.dump(args, g)
                
                # save all trainstate objects
                for i in range(3):
                    new_outfile = all_savemodel_filenames[i].replace('.pkl','_BROKEN.pkl')
                    with open(new_outfile, 'wb') as g:
                        model_state_dict = flax.serialization.to_state_dict(all_trainstates[i])
                        pickle.dump(model_state_dict, g)
                
                # save the batch
                batch_samples.to_csv(f'{args.logfile_dir}/NAN-BATCH.tsv',
                                     sep='\t')
                
                # one last recording to tensorboard
                write_optional_outputs_during_training(writer_obj = writer, 
                                                       all_trainstates = all_trainstates,
                                                       global_step = batch_epoch_idx, 
                                                       dict_of_values = train_metrics, 
                                                       interms_for_tboard = args.interms_for_tboard, 
                                                       write_histograms_flag = False)
                
                # save the batch elements itself
                for i, mat in enumerate( batch[:-1] ):
                    with open( f'{args.out_arrs_dir}/NAN-BATCH_matrix{i}.npy','wb' ) as g:
                        np.save(g, mat)
                    
                raise RuntimeError( ('NaN loss detected; saved intermediates '+
                                    'and quit training') )
            
            
#__4___8__12: batch level (three tabs)
            ### add to recorded metrics for this epoch
            weight = args.batch_size / len(training_dset)
            
            ave_epoch_train_loss += train_metrics['batch_loss'] * weight
            ave_epoch_train_perpl += train_metrics['batch_ave_perpl'] * weight
            
            # may or may not have accuracy measured
            if have_acc:
                ave_epoch_train_acc += train_metrics['batch_ave_acc'] * weight
            
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
            
               
            # record the CPU+system and wall-clock (real) time
            batch_real_end = wall_clock_time()
            batch_cpu_end = process_time()
            write_times(cpu_start = batch_cpu_start, 
                        cpu_end = batch_cpu_end, 
                        real_start = batch_real_start, 
                        real_end = batch_real_end, 
                        tag = 'Process one training batch', 
                        step = batch_epoch_idx, 
                        writer_obj = writer)
            
            del batch_cpu_start, batch_cpu_end, batch_real_start, batch_real_end
        
        
#__4___8: epoch level (two tabs)
        ##############################################################
        ### 3.3: also check current performance on held-out test set #
        ##############################################################
        # Note: it's possible to output intermediates for these points too;
        # but right now, that's not collected
        ave_epoch_test_loss = 0
        ave_epoch_test_perpl = 0
        if have_acc:
            ave_epoch_test_acc = 0
        
        for batch_idx, batch in enumerate(test_dl):
            # unpack briefly to get max len and number of samples in the 
            #   batch; place in some bin (this controls how many jit 
            #   compilations you do)
            if have_full_length_alignments:
                batch_max_seqlen = jitted_determine_seqlen_bin(batch = batch)
                batch_max_seqlen = batch_max_seqlen.item()
                batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch)
                batch_max_alignlen = batch_max_alignlen.item()
                
                # I've had so much trouble with this ugh
                if args.use_scan_fns:
                    err = (f'batch_max_alignlen (not including bos) is: '+
                           f'{batch_max_alignlen - 1}'+
                           f', which is not divisible by length for scan '+
                           f'({args.chunk_length})')
                    assert (batch_max_alignlen - 1) % args.chunk_length == 0, err
            
            elif not have_full_length_alignments:
                batch_max_seqlen = None
                batch_max_alignlen = None
            
            eval_metrics = eval_fn_jitted(batch=batch, 
                                          all_trainstates=all_trainstates,
                                          max_seq_len=batch_max_seqlen.item(),
                                          max_align_len=batch_max_alignlen.item())
            
#__4___8__12: batch level (three tabs)
            ### add to total loss for this epoch; weight by number of
            ###   samples/valid tokens in this batch
            weight = args.batch_size / len(test_dset)
            ave_epoch_test_loss += eval_metrics['loss'] * weight
            ave_epoch_test_perpl += jnp.mean( eval_metrics['perplexity_perSamp'] ) * weight
            
            if have_acc:
                ave_epoch_test_acc += jnp.mean( eval_metrics['acc_perSamp'] ) * weight
        
            
#__4___8: epoch level (two tabs) 
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
        
        # accuracy, if applicable
        if have_acc:
            writer.add_scalar(tag='Accuracy/training set',
                              scalar_value=ave_epoch_train_acc.item(), 
                              global_step=epoch_idx)
                
            writer.add_scalar(tag='Accuracy/test set', 
                              scalar_value=ave_epoch_test_acc.item(), 
                              global_step=epoch_idx)
            
            
#__4___8: epoch level (two tabs) 
        ##########################################################
        ### 3.5: if this is the best epoch TEST loss,            #
        ###      save the model params and args for later eval   #
        ##########################################################
        if ave_epoch_test_loss < best_test_loss:
            ### !!!!!!!!!!!!!!! BELOW IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
            note = 'best_epoch'
            ### !!!!!!!!!!!!!!! ABOVE IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
            
            with open(args.logfile_name,'a') as g:
                g.write((f'New best test loss at epoch {epoch_idx}: ') +
                        (f'{ave_epoch_test_loss}\n'))
            
            # update "best" recordings
            best_test_loss = ave_epoch_test_loss
            best_trainstates = all_trainstates
            best_epoch = epoch_idx
            
            if have_acc:
                best_test_acc = ave_epoch_test_acc
            
            # save models to regular python pickles too (in case training is 
            #   interrupted)
            for i in range(3):
                with open(f'{all_savemodel_filenames[i]}', 'wb') as g:
                    model_state_dict = flax.serialization.to_state_dict(all_trainstates[i])
                    pickle.dump(model_state_dict, g)
                    
            if not have_full_length_alignments:
                notice = ( 'un-transform the logits and write lambda, mu, '+
                            '(+r,x,y) somewhere easy to check' )
                raise NotImplementedError(notice)
            
            
        ### !!!!!!!!!!!!!!! BELOW IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
        else:
            note = 'none'
        ### !!!!!!!!!!!!!!! ABOVE IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
        

#__4___8: epoch level (two tabs) 
        ### !!!!!!!!!!!!!!! BELOW IS FOR DEBUG-ONLY !!!!!!!!!!!!!!!
        ### also write losses to text file
        with open(f'{args.logfile_dir}/LOSSES.tsv','a') as g:
            g.write( (f'{epoch_idx}' + '\t' +
                      f'train_dset' + '\t' +
                      f'DSET_AVE' + '\t' +
                      f"{ave_epoch_train_loss.item()}" + '\t' +
                      f"{ave_epoch_train_perpl.item()}" + '\t' +
                      f"{np.exp( ave_epoch_train_loss.item() )}" + '\t' +
                      f"{note}" +'\n') 
                    )
            
            g.write( (f'{epoch_idx}' + '\t' +
                      f'test_dset' + '\t' +
                      f'DSET_AVE' + '\t' +
                      f"{ave_epoch_test_loss.item()}" + '\t' +
                      f"{ave_epoch_test_perpl.item()}" + '\t' +
                      f"{np.exp( ave_epoch_test_loss.item() )}" + '\t' +
                      f"{note}" +'\n') 
                    )
        ### !!!!!!!!!!!!!!! ABOVE IS FOR DEBUG-ONLY !!!!!!!!!!!!!!! 
            
        
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
            
            write_times(cpu_start = epoch_cpu_start, 
                        cpu_end = epoch_cpu_end, 
                        real_start = epoch_real_start, 
                        real_end = epoch_real_end, 
                        tag = 'Process one epoch', 
                        step = epoch_idx, 
                        writer_obj = writer)
            
            del epoch_cpu_start, epoch_cpu_end
            
            # save the trainstates for later use
            best_trainstates = all_trainstates
            
            # rage quit
            break
        

#__4___8: epoch level (two tabs) 
        ### before next epoch, do this stuff
        # remember this epoch's loss for next iteration
        prev_test_loss = ave_epoch_test_loss
        
        # record time spent at this epoch
        epoch_real_end = wall_clock_time()
        epoch_cpu_end = process_time()
        
        write_times(cpu_start = epoch_cpu_start, 
                    cpu_end = epoch_cpu_end, 
                    real_start = epoch_real_start, 
                    real_end = epoch_real_end, 
                    tag = 'Process one epoch', 
                    step = epoch_idx, 
                    writer_obj = writer)
        
        del epoch_cpu_start, epoch_cpu_end
    
    
    ###########################################################################
    ### 4: POST-TRAINING ACTIONS   ############################################
    ###########################################################################
    print(f'4: post-training actions')
    # write to logfile
    with open(args.logfile_name,'a') as g:
        g.write('\n')
        g.write(f'4: post-training actions\n')
    
    post_training_real_start = wall_clock_time()
    post_training_cpu_start = process_time()
    
    # don't accidentally use old trainstates or eval fn
    del all_trainstates, eval_fn_jitted
    
    
    ### write to output logfile
    with open(args.logfile_name,'a') as g:
        # if early stopping was never triggered, record results at last epoch
        if early_stopping_counter != args.patience:
            g.write(f'Regular stopping after {epoch_idx} full epochs:\n\n')
        
        if not have_full_length_alignments:
            notice = ( 'un-transform the logits and write lambda, mu, '+
                        '(+r,x,y) to this file' )
            raise NotImplementedError(notice)
        
        # finish up logfile, regardless of early stopping or not
        g.write(f'Epoch with lowest average test loss ("best epoch"): {best_epoch}\n\n')
        g.write(f'RE-EVALUATING ALL DATA WITH BEST PARAMS:\n\n')
    
    del epoch_idx
    
    
    raise NotImplementedError('Make jitted eval function')
    raise NotImplementedError('Make a new final eval wrapper function')
    
    """
    ### jit-compile a different version of eval function for final eval    
    # if both are transformer models, will have extra arguments for eval funciton
    # TODO: how to handle if only one is transformer?
    extra_args_for_eval = dict()
    if (args.anc_model_type == 'Transformer' and args.desc_model_type == 'Transformer'):
        flag = (args.anc_enc_config.get('output_attn_weights',False) or 
                args.desc_dec_config.get('output_attn_weights',False))
        extra_args_for_eval['output_attn_weights'] = flag
        
    
    # new parted_eval_fn 
    parted_eval_fn = partial(eval_one_batch,
                             all_model_instances = all_model_instances,
                             length_for_scan = args.chunk_length,
                             which_alignment_states_to_encode = args.which_alignment_states_to_encode, 
                             interms_for_tboard = args.interms_for_tboard, 
                             seq_padding_idx = args.seq_padding_idx,
                             norm_loss_by = args.pred_config['norm_loss_by'],
                             have_time_values = have_time_values,
                             extra_args_for_eval = extra_args_for_eval)
    del extra_args_for_eval
    
    
    # new jit compiled eval function to use for final evaluation
    eval_fn_jitted = jax.jit(parted_eval_fn, 
                             static_argnames = ['max_seq_len',
                                                'max_align_len'])
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
                                             jitted_determine_seqlen_bin = jitted_determine_seqlen_bin,
                                             jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                                             eval_fn_jitted = eval_fn_jitted,
                    # [when do I use this?]  out_alph_size = args.full_alphabet_size,
                                             save_arrs = args.save_arrs,
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
        g.write(f'SCORING ALL TEST SEQS\n\n')
        
    # output_attn_weights also controlled by cond1 and cond2
    test_summary_stats = final_eval_wrapper(dataloader = test_dl, 
                                             dataset = test_dset, 
                                             best_trainstates = best_trainstates, 
                                             jitted_determine_seqlen_bin = jitted_determine_seqlen_bin,
                                             jitted_determine_alignlen_bin = jitted_determine_alignlen_bin,
                                             eval_fn_jitted = eval_fn_jitted,
                    # [when do I use this?]  out_alph_size = args.full_alphabet_size, 
                                             save_arrs = args.save_arrs,
                                             interms_for_tboard = args.interms_for_tboard, 
                                             logfile_dir = args.logfile_dir,
                                             out_arrs_dir = args.out_arrs_dir,
                                             outfile_prefix = f'test-set',
                                             tboard_writer = writer)
    """
    
    
    ###########################################
    ### update the logfile with final losses  #
    ###########################################
    with open(args.logfile_name,'a') as g:
        g.write(f'TRAIN SET:\n')
        g.write(f'==========\n')
        for key, val in train_summary_stats.items():
            g.write(f'{key}: {val}\n')
        g.write('\n')
        
        g.write(f'TEST SET:\n')
        g.write(f'==========\n')
        for key, val in test_summary_stats.items():
            g.write(f'{key}: {val}\n')
            
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
    