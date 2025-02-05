#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 00:50:39 2023

@author: annabel_large

ABOUT:
======
train and eval functions for one batch of data


THIS USES A SCAN IMPLEMENTATION

"""
# regular python
import numpy as np
from collections.abc import MutableMapping
import pickle
import math
from functools import partial

# flax, jax, and optax
import jax
import jax.numpy as jnp
from jax import config
from flax import linen as nn
import optax

from models.modeling_utils.concatenation_fns import extract_embs
from utils.sequence_length_helpers import selective_squeeze


###############################################################################
### TRAIN ON ONE BATCH    #####################################################
###############################################################################
def train_one_batch(batch, 
                    training_rngkey,
                    all_model_instances,
                    all_trainstates,  
                    max_seq_len,
                    max_align_len,
                    which_alignment_states_to_encode,
                    interms_for_tboard, 
                    have_time_values: bool,
                    length_for_scan = None,
                    norm_loss_by = 'desc_len',
                    seq_padding_idx = 0,
                    align_idx_padding = -9):
    """
    *** NOTE: THIS IS HARD-CODED FOR PROTEINS!!! ***
    
    Jit-able function to apply the model to one batch of samples, evaluate loss
    and collect gradients, then update model parameters
    
    regular inputs:
        > batch: batch from a pytorch dataloader
        > training_rngkey: the rng key
        > all_trainstates: the models + parameters
    
    static inputs:
        > all_model_instances: contains methods specific to architectures
        > max_seq_len: max length of unaligned seqs matrix (used to control 
                       number of jit-compiled versions of this function)
        > max_align_len: max length of alignment matrix (used to control 
                         number of jit-compiled versions of this function)
        > length_for_scan: max length of alignment inputs to calculate loss 
                           for, at any given time in jax.lax.scan      
          > if None, then auto-infer from max_align_len
        > which_alignment_states_to_encode: ['prev_OH', 'both_cat', None]  
          (behavior depends on which prediction head is used; determined in 
           train_cli)
        > interms_for_tboard: decide whether or not to output intermediate 
                             histograms and scalars
        > norm_loss_by: what length to normalize losses by
        > seq_padding_idx: usually zero
        > align_idx_padding: usually -9
    
    outputs:
        > metrics_outputs: dictionary of metrics and outputs                                  
        
    """
    # if you're not chunking, length_for_scan is None
    if (length_for_scan is None):
        length_for_scan = (max_align_len - 1)
            
    
    #########################
    ### UNPACK FLAGS, FNS   #
    #########################
    # booleans for determining which sowed outputs to write to
    #   tensorboard
    encoder_sow_outputs = interms_for_tboard.get('encoder_sow_outputs', False)
    decoder_sow_outputs = interms_for_tboard.get('decoder_sow_outputs', False)
    finalpred_sow_outputs = interms_for_tboard.get('finalpred_sow_outputs',False)
    save_gradients = interms_for_tboard.get('gradients',False)
    save_updates = interms_for_tboard.get('optimizer',False)
    del interms_for_tboard
    
    
    ##################################
    ### UNPACK THE INPUTS, PREPROC   #
    ##################################
    ### unpack the trainstates
    encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
    del all_trainstates
    
    ### unpack all_model_instances, which contain helpers for applying
    ###  the models
    encoder_instance, decoder_instance, finalpred_instance = all_model_instances
    del all_model_instances
    
    ### clip to max lengths
    batch_unaligned_seqs, batch_aligned_mats, t_array, _ = batch
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :]
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    del batch, batch_unaligned_seqs, batch_aligned_mats
    
    
    ### unpack sequences
    # unaligned sequences used in __call__; final size is (B, max_seq_len)
    anc_seqs = clipped_unaligned_seqs[:,:,0]
    desc_seqs = clipped_unaligned_seqs[:,:,1]
    del clipped_unaligned_seqs
    
    # precomputed alignment indices; final size is (B, max_align_len-1, 2)
    # don't include last token, since it's not used to predict any valid input
    align_idxes = clipped_aligned_mats[:,:-1,[-2,-1]]
    
    # true alignments used in evaluate_loss; shift view of length dim up by
    # one, since you're never predicting the probability of <bos>
    # for feedforward: final size is (B, max_align_len-1)
    # for tkf92: final size is (B, max_align_len-1, 2)
    true_out = selective_squeeze( clipped_aligned_mats[:,1:,:-2] )
    
    
    ### find length to normalize by
    ###   (B, 1)
    length_for_normalization = finalpred_instance.get_length_for_normalization(true_out = true_out,
                                                            norm_loss_by = norm_loss_by,
                                                            seq_padding_idx = seq_padding_idx,
                                                            gap_tok = 43)
    
    
    ### encode all alignment position's state
    # one-hot encoded for feedforward head: 
    # input is (B, max_align_len)
    # output is (B, max_align_len - 1, 6)
    if which_alignment_states_to_encode == 'prev_OH':
        to_encode = selective_squeeze( clipped_aligned_mats[:, :-1, :-2] )
        alignment_state = finalpred_instance.encode_states(arr = to_encode)
    
    # categorically encoded for tkf92 head: 
    # input is (B, max_align_len, 2)
    # output is (B, max_align_len - 1, 2)
    elif which_alignment_states_to_encode == 'both_cat':
        raw_alignment_state = finalpred_instance.encode_states(arr = clipped_aligned_mats[:, :, :-2])[:,:,None]
        
        alignment_state = jnp.concatenate( [ raw_alignment_state[:, :-1, :],
                                             raw_alignment_state[:,  1:, :] ],
                                           axis = -1)
        del raw_alignment_state
        
    # (B, max_align_len - 1)
    elif which_alignment_states_to_encode == None:
        alignment_state = None
    
    del clipped_aligned_mats
    
    
    ### produce new keys for each network
    all_keys = jax.random.split(training_rngkey, num=4)
    training_rngkey, enc_key, dec_key, finalpred_key = all_keys
    del all_keys
    
    ############################################
    ### APPLY MODEL, EVALUATE LOSS AND GRADS   #
    ############################################
    def apply_model(encoder_params, decoder_params, finalpred_params):
        ### embed with ancestor encoder
        # anc_embeddings is (B, max_seq_len, H)
        anc_embeddings, embeddings_aux_dict = encoder_instance.apply_seq_embedder_in_training(seqs = anc_seqs,
                                                               rng_key = enc_key,
                                                               params_for_apply = encoder_params,
                                                               seq_emb_trainstate = encoder_trainstate,
                                                               sow_outputs = encoder_sow_outputs)
        
        
        ### embed with descendant decoder
        # desc_embeddings is (B, max_seq_len, H)
        desc_embeddings, to_add = decoder_instance.apply_seq_embedder_in_training(seqs = desc_seqs,
                                                   rng_key = dec_key,
                                                   params_for_apply = decoder_params,
                                                   seq_emb_trainstate = decoder_trainstate,
                                                   sow_outputs = decoder_sow_outputs)
        
        # at minimum, embeddings_aux_dict contains:
        #     - anc_aux
        #     - anc_layer_metrics
        #     - desc_layer_metrics
        
        # could also contain
        #     - anc_attn_weights (for transformers)
        #     - desc_attn_weights (for transformers)
        #     - anc_carry (for LSTMs)
        #     - desc_carry (for LSTMs)
        embeddings_aux_dict = {**embeddings_aux_dict, **to_add}
        del to_add
        
        
        ### apply scan over alignment length
        start_indices = jnp.arange(0, (max_align_len - 1), length_for_scan)
        def scan_during_training(carry_loss, start):
            scan_finalpred_key = jax.random.fold_in(finalpred_key, start)
            
            # slice along length
            # (B, max_align_len-1, 2) -> (B, length_for_scan, 2)
            align_idxes_scan_clipped = jax.lax.dynamic_slice_in_dim(operand = align_idxes, 
                                              start_index = start, 
                                              slice_size = length_for_scan,
                                              axis=1)
            
            # run forward pass + loss as normal
            # both outputs are (B, length_for_scan, H)
            reprs_for_alignment, output_padding_mask = extract_embs(anc_encoded = anc_embeddings, 
                                                                    desc_encoded = desc_embeddings,
                                                                    idx_lst = align_idxes_scan_clipped,
                                                                    align_idx_padding = align_idx_padding)
            
            # handle alignment path, depending on the model
            if which_alignment_states_to_encode:
                alignment_state_scan_clipped = jax.lax.dynamic_slice_in_dim(operand = alignment_state, 
                                                  start_index = start, 
                                                  slice_size = length_for_scan,
                                                  axis=1)
                
                if which_alignment_states_to_encode == 'prev_OH':
                    reprs_for_alignment.append(alignment_state_scan_clipped)
            
            else:
                alignment_state_scan_clipped = None
                
            # __call__ method on the chunk
            forward_pass_outputs, _ = finalpred_trainstate.apply_fn(variables = finalpred_params,
                                                                    datamat_lst = reprs_for_alignment,
                                                                    t_array = t_array,
                                                                    padding_mask = output_padding_mask,
                                                                    training = True,
                                                                    sow_intermediates = finalpred_sow_outputs,
                                                                    mutable=['histograms','scalars'] if finalpred_sow_outputs else [],
                                                                    rngs={'dropout': scan_finalpred_key})
            
            ### TODO: figure out how to handle this intermediates dictionary
            ###   over iterations of jax.lax.scan
            # pred_layer_metrics = {'histograms': pred_sow_dict.get( 'histograms', dict() ),
            #                       'scalars': pred_sow_dict.get( 'scalars', dict() )
            #                       }
            
            # evaluate loglike on the chunk
            true_out_scan_clipped = jax.lax.dynamic_slice_in_dim(operand = true_out, 
                                              start_index = start, 
                                              slice_size = length_for_scan,
                                              axis=1)
            out = finalpred_instance.neg_loglike_in_scan_fn(forward_pass_outputs = forward_pass_outputs,
                                                            true_out = true_out_scan_clipped,
                                                            alignment_state = alignment_state_scan_clipped,
                                                            seq_padding_idx = 0)
            sum_neg_logP_for_chunk, intermed_dict_for_chunk = out
            del out
            
            # some metrics need output from forward pass, so calculate those here
            metrics_in_scan = finalpred_instance.compile_metrics_in_scan(forward_pass_outputs = forward_pass_outputs, 
                                                      true_out = true_out_scan_clipped, 
                                                      seq_padding_idx = 0)
            intermed_dict_for_chunk = {**intermed_dict_for_chunk, **metrics_in_scan}
            intermed_dict_for_chunk['forward_pass_outputs'] = forward_pass_outputs
            
            # add to previous iteration
            carry_loss = carry_loss + sum_neg_logP_for_chunk
            
            return carry_loss, intermed_dict_for_chunk
        
        
        ### use scanned version of function
        # output loss should be (T, B), even if no time provided
        num_timepoints = len(t_array) if have_time_values else 1
        scan_fn_outputs = jax.lax.scan(f = scan_during_training,
                           init = jnp.zeros( (num_timepoints, true_out.shape[0]) ),
                           xs = start_indices,
                           length = start_indices.shape[0])
        
        
        ### normalize sum_neg_logP by desired length and take the mean
        ###   also concat the intermed dict
        out = finalpred_instance.evaluate_loss_after_scan(scan_fn_outputs = scan_fn_outputs,
                                                 length_for_normalization = length_for_normalization,
                                                 seq_padding_idx = seq_padding_idx,
                                                 t_array = t_array)
        loss, aux_dict = out
        del out, scan_fn_outputs
        
        
        ### TODO: concat values from pred_layer_metrics, if there's any
        ###   so that code runs, just return empty dictionary for now
        pred_layer_metrics = dict()
        
        
        ### manage the dictionary of auxilary values
        aux_dict['embeddings_aux_dict'] = embeddings_aux_dict
        aux_dict['pred_layer_metrics'] = pred_layer_metrics
        return (loss, aux_dict)
    
    
    ### set up the grad functions, based on above loss function
    grad_fn = jax.value_and_grad(apply_model, argnums=[0,1,2], has_aux=True)
    
    (batch_loss, aux_dict), all_grads = grad_fn(encoder_trainstate.params, 
                                                decoder_trainstate.params, 
                                                finalpred_trainstate.params)
    
    enc_gradient, dec_gradient, finalpred_gradient = all_grads
    del all_grads
    
    # aux_dict contains:
    #     - intermediate metrics collected during scan
    #       > for feedforward, this is pred_outputs
    #       > nothing for tkf92 yet
    #     - intermediates from pred_layer_metrics
    
    # all will have dummy axis inserted by jax.lax.scan, such that their 
    #   dimensions are (num_scan_iters, ..., chunk_size, ...)
    
    ### evaluate metrics
    metrics_dict = finalpred_instance.compile_metrics(true_out = true_out,
                                                      loss = batch_loss,
                                                      loss_fn_dict = aux_dict,
                                                      seq_padding_idx = seq_padding_idx)
    
    batch_ave_perpl = jnp.mean(metrics_dict['perplexity_perSamp'])
    
    if 'acc_perSamp' in metrics_dict.keys():
        batch_ave_acc = jnp.mean(metrics_dict['acc_perSamp'])
    else:
        batch_ave_acc = None
        
    
    # ### DEBUG: output gradients to pickles
    # with open('ENC_GRADIENT_DICT.pkl','wb') as g:
    #     pickle.dump(enc_gradient, g)
    
    
    # with open('DEC_GRADIENT_DICT.pkl','wb') as g:
    #     pickle.dump(dec_gradient, g)
    
    
    # with open('finalpred_GRADIENT_DICT.pkl','wb') as g:
    #     pickle.dump(finalpred_gradient, g)
    
    
    ###########################
    ### RECORD UPDATES MADE   #
    ###########################
    ### get new updates and optimizer states
    encoder_updates, new_encoder_opt_state = encoder_trainstate.tx.update(enc_gradient, 
                                                                          encoder_trainstate.opt_state, 
                                                                          encoder_trainstate.params)
    
    decoder_updates, new_decoder_opt_state  = decoder_trainstate.tx.update(dec_gradient, 
                                                                           decoder_trainstate.opt_state, 
                                                                           decoder_trainstate.params)
    
    finalpred_updates, new_finalpred_opt_state  = finalpred_trainstate.tx.update(finalpred_gradient, 
                                                                                 finalpred_trainstate.opt_state, 
                                                                                 finalpred_trainstate.params)
    
    ### apply updates to parameter, trainstate object
    # wrapper for encoder, in case I ever use batch norm
    new_encoder_trainstate = encoder_instance.update_seq_embedder_tstate(tstate = encoder_trainstate,
                                                        new_opt_state = new_encoder_opt_state,
                                                        optim_updates = encoder_updates)
    del new_encoder_opt_state
    
    # standard update for decoder
    new_decoder_params = optax.apply_updates(decoder_trainstate.params, 
                                             decoder_updates)
    new_decoder_trainstate = decoder_trainstate.replace(params = new_decoder_params,
                                                        opt_state = new_decoder_opt_state)
    del new_decoder_opt_state
    
    # standard update for prediction head
    new_finalpred_params = optax.apply_updates(finalpred_trainstate.params, 
                                               finalpred_updates)
    new_finalpred_trainstate = finalpred_trainstate.replace(params = new_finalpred_params,
                                                            opt_state = new_finalpred_opt_state)
    del new_finalpred_opt_state
    
    
    ###############
    ### OUTPUTS   #
    ###############
    # repackage the new trainstates
    updated_trainstates = (new_encoder_trainstate, 
                           new_decoder_trainstate, 
                           new_finalpred_trainstate)
    
    
    ### always returned
    out_dict = {'anc_aux': aux_dict['embeddings_aux_dict']['anc_aux'],
                'batch_loss': batch_loss,
                'batch_ave_acc': batch_ave_acc,
                'batch_ave_perpl': batch_ave_perpl
                }
    
    ### controlled by boolean flag
    def save_to_out_dict(value_to_save, flag, varname_to_write):
        if flag:
            out_dict[varname_to_write] = value_to_save
            
    # intermediate values captured during training
    save_to_out_dict(value_to_save = aux_dict['embeddings_aux_dict']['anc_layer_metrics'], 
                           flag = encoder_sow_outputs, 
                           varname_to_write = 'anc_layer_metrics')
    
    save_to_out_dict(value_to_save = aux_dict['embeddings_aux_dict']['desc_layer_metrics'], 
                           flag = decoder_sow_outputs, 
                           varname_to_write = 'desc_layer_metrics')
    
    save_to_out_dict(value_to_save = aux_dict['pred_layer_metrics'], 
                           flag = finalpred_sow_outputs, 
                           varname_to_write = 'pred_layer_metrics')
    
    # gradients
    for (varname, grad) in [('enc_gradient', enc_gradient),
                            ('dec_gradient', dec_gradient),
                            ('finalpred_gradient', finalpred_gradient)]:
        save_to_out_dict(value_to_save = grad,
                               flag = save_gradients,
                               varname_to_write = varname)
    # updates
    for (varname, grad) in [('encoder_updates', encoder_updates),
                            ('decoder_updates', decoder_updates),
                            ('finalpred_updates', finalpred_updates)]:
        save_to_out_dict(value_to_save = grad,
                               flag = save_updates,
                               varname_to_write = varname)
    
    # always returned from out_dict:
    #     - anc_aux (structure varies depending on model)
    #     - batch_loss; float
    #     - batch_ave_perpl; float
    #     - encoder_updates
    #     - decoder_updates
    #     - finalpred_updates
    
    # returned if using a feedforward prediction head:
    #     - batch_ave_acc; float (if applicable)
    
    # returned if flag active:
    #     - anc_layer_metrics
    #     - desc_layer_metrics
    #     - pred_layer_metrics
    #     - enc_gradient
    #     - dec_gradient
    #     - finalpred_gradient 
    #     - encoder_updates
    #     - decoder_updates
    #     - finalpred_updates

    return (out_dict, updated_trainstates)




###############################################################################
### EVAL ON ONE BATCH    ######################################################
###############################################################################
def eval_one_batch(batch, 
                   all_trainstates, 
                   all_model_instances,  
                   max_seq_len,
                   max_align_len,
                   which_alignment_states_to_encode, 
                   interms_for_tboard, 
                   have_time_values: bool,
                   length_for_scan = None,
                   seq_padding_idx = 0,
                   align_idx_padding = -9,
                   norm_loss_by = 'desc_len',
                   extra_args_for_eval: dict = dict()):
    """
    JIT-able function to evaluate on a batch of samples
    
    regular inputs:
        > batch: batch from a pytorch dataloader
        > all_trainstates: the models + parameters
    
    static inputs:
        (most given above by train_one_batch)
        > extra_args_for_eval: extra inputs for custom eval functions
    
    outputs:
        > metrics_outputs: dictionary of metrics and outputs
            
    """
    #########################
    ### UNPACK FLAGS, FNS   #
    #########################
    # booleans for determining which sowed outputs to write to
    #   tensorboard
    encoder_sow_outputs = interms_for_tboard.get('encoder_sow_outputs', False)
    decoder_sow_outputs = interms_for_tboard.get('decoder_sow_outputs', False)
    finalpred_sow_outputs = interms_for_tboard.get('finalpred_sow_outputs', False)
    
    # booleans for determining which intermediate arrays to return
    return_anc_embs = interms_for_tboard.get('ancestor_embeddings', False)
    return_desc_embs = interms_for_tboard.get('descendant_embeddings', False)
    return_forward_pass_outputs = interms_for_tboard.get('forward_pass_outputs', False)
    return_final_logprobs = interms_for_tboard.get('final_logprobs', False)
    del interms_for_tboard
    
    
    ##################################
    ### UNPACK THE INPUTS, PREPROC   #
    ##################################
    ### the trainstates
    encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
    del all_trainstates
    
    
    ### unpack all_model_instances, which contain helpers for applying
    ###  the models
    encoder_instance, decoder_instance, finalpred_instance = all_model_instances
    del all_model_instances
    
    
    ### clip to max lengths
    batch_unaligned_seqs, batch_aligned_mats, t_array, _ = batch
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :]
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    del batch, batch_unaligned_seqs, batch_aligned_mats
    
    
    ### unpack sequences
    # unaligned sequences used in __call__; final size is (B, max_seq_len)
    anc_seqs = clipped_unaligned_seqs[:,:,0]
    desc_seqs = clipped_unaligned_seqs[:,:,1]
    del clipped_unaligned_seqs
    
    # precomputed alignment indices; final size is (B, max_align_len-1, 2)
    # don't include last token, since it's not used to predict any valid input
    align_idxes = clipped_aligned_mats[:,:-1,[-2,-1]]
    
    # true alignments used in evaluate_loss; shift view of length dim up by
    # one, since you're never predicting the probability of <bos>
    #
    # for feedforward: final size is (B, max_align_len-1, 1)
    # for tkf92: final size is (B, max_align_len-1, 2)
    true_out = selective_squeeze( clipped_aligned_mats[:,1:,:-2] )
    
    ### find length to normalize by
    ###   (B, 1)
    length_for_normalization = finalpred_instance.get_length_for_normalization(true_out = true_out,
                                                            norm_loss_by = norm_loss_by,
                                                            seq_padding_idx = seq_padding_idx,
                                                            gap_tok = 43)
    
    
    ### encode all alignment position's state
    # one-hot encoded for feedforward head: 
    # input is (B, max_align_len, 6)
    # output is (B, max_align_len - 1, 6)
    if which_alignment_states_to_encode == 'prev_OH':
        to_encode = selective_squeeze( clipped_aligned_mats[:, :-1, :-2] )
        alignment_state = finalpred_instance.encode_states(arr = to_encode)
    
    # categorically encoded for tkf92 head: 
    # input is (B, max_align_len, 1)
    # output is (B, max_align_len - 1, 2)
    elif which_alignment_states_to_encode == 'both_cat':
        raw_alignment_state = finalpred_instance.encode_states(arr = clipped_aligned_mats[:, :, :-2])[:,:,None]
        alignment_state = jnp.concatenate( [ raw_alignment_state[:, :-1],
                                             raw_alignment_state[:,  1:] ],
                                           axis = -1)
        del raw_alignment_state
        
    elif which_alignment_states_to_encode == None:
        alignment_state = None
    
    del clipped_aligned_mats
    
    
    #######################
    ### Apply the model   #
    #######################
    ### embed with ancestor encoder
    anc_embeddings, embeddings_aux_dict = encoder_instance.apply_seq_embedder_in_eval(seqs = anc_seqs,
                                                           final_trainstate = encoder_trainstate,
                                                           sow_outputs = encoder_sow_outputs,
                                                           extra_args_for_eval = extra_args_for_eval)
    
    ### embed with descendant decoder
    desc_embeddings, to_add = decoder_instance.apply_seq_embedder_in_eval(seqs = desc_seqs,
                                                           final_trainstate = decoder_trainstate,
                                                           sow_outputs = decoder_sow_outputs,
                                                           extra_args_for_eval = extra_args_for_eval)
    
    # at minimum, embeddings_aux_dict contains:
    #     - anc_aux
    #     - anc_layer_metrics
    #     - desc_layer_metrics
    
    # could also contain
    #     - anc_attn_weights (for transformers)
    #     - desc_attn_weights (for transformers)
    #     - anc_carry (for LSTMs)
    #     - desc_carry (for LSTMs)
    embeddings_aux_dict = {**embeddings_aux_dict, **to_add}
    del to_add
    
    
    
    ### apply scan over alignment length
    # if you're not chunking, length_for_scan is None
    if (length_for_scan is None):
        length_for_scan = (max_align_len - 1)
        
    start_indices = jnp.arange(0, (max_align_len - 1), length_for_scan)
    def scan_during_eval(carry_loss, start):
        # slice along length
        # (B, max_align_len-1, 2+i) -> (B, length_for_scan, 2+i)
        # i=1 for feedforward head, i=2 for tkf92 head
        align_idxes_scan_clipped = jax.lax.dynamic_slice_in_dim(operand = align_idxes, 
                                          start_index = start, 
                                          slice_size = length_for_scan,
                                          axis=1)
        
        # run forward pass + loss as normal
        # both outputs are (B, length_for_scan, H)
        reprs_for_alignment, output_padding_mask = extract_embs(anc_encoded = anc_embeddings, 
                                                                desc_encoded = desc_embeddings,
                                                                idx_lst = align_idxes_scan_clipped,
                                                                align_idx_padding = align_idx_padding)
        
        # handle alignment path, depending on the model
        if which_alignment_states_to_encode:
            alignment_state_scan_clipped = jax.lax.dynamic_slice_in_dim(operand = alignment_state, 
                                              start_index = start, 
                                              slice_size = length_for_scan,
                                              axis=1)
            
            if which_alignment_states_to_encode == 'prev_OH':
                reprs_for_alignment.append(alignment_state_scan_clipped)
        
        else:
            alignment_state_scan_clipped = None
            
        # __call__ method on the chunk
        forward_pass_outputs, pred_sow_dict = finalpred_trainstate.apply_fn(variables = finalpred_trainstate.params,
                                                                            datamat_lst = reprs_for_alignment,
                                                                            t_array = t_array,
                                                                            padding_mask = output_padding_mask,
                                                                            training = False,
                                                                            sow_intermediates = finalpred_sow_outputs,
                                                                            mutable=['histograms','scalars'] if finalpred_sow_outputs else [],
                                                                            )
        
        ### TODO: figure out how to handle this intermediates dictionary
        ###   over iterations of jax.lax.scan
        # pred_layer_metrics = {'histograms': pred_sow_dict.get( 'histograms', dict() ),
        #                       'scalars': pred_sow_dict.get( 'scalars', dict() )
        #                       }
        
        # evaluate_loss_in_scan on the chunk
        true_out_scan_clipped = jax.lax.dynamic_slice_in_dim(operand = true_out, 
                                          start_index = start, 
                                          slice_size = length_for_scan,
                                          axis=1)
        out = finalpred_instance.neg_loglike_in_scan_fn(forward_pass_outputs = forward_pass_outputs,
                                                       true_out = true_out_scan_clipped,
                                                       alignment_state = alignment_state_scan_clipped,
                                                       seq_padding_idx = 0)
        sum_neg_logP_for_chunk, intermed_dict_for_chunk = out
        
        del out
        
        # some metrics need output from forward pass, so calculate those here
        metrics_in_scan = finalpred_instance.compile_metrics_in_scan(forward_pass_outputs = forward_pass_outputs, 
                                                  true_out = true_out_scan_clipped, 
                                                  seq_padding_idx = 0)

        intermed_dict_for_chunk = {**intermed_dict_for_chunk, **metrics_in_scan}
        
        # possibly return forward pass outputs
        if return_forward_pass_outputs:
            for key in forward_pass_outputs:
                if key.startswith('FPO_'):
                    intermed_dict_for_chunk[key] = forward_pass_outputs[key] 
        
        # add to previous iteration
        carry_loss = carry_loss + sum_neg_logP_for_chunk
        
        return carry_loss, intermed_dict_for_chunk
    
    ### use scanned version of function
    num_timepoints = len(t_array) if have_time_values else 1
    scan_fn_outputs = jax.lax.scan(f = scan_during_eval,
                                   init = jnp.zeros( (num_timepoints, true_out.shape[0]) ),
                                   xs = start_indices,
                                   length = start_indices.shape[0])
    
    
    ### normalize sum_neg_logP by desired length and take the mean
    ###   also concat the intermed dict
    out = finalpred_instance.evaluate_loss_after_scan(scan_fn_outputs = scan_fn_outputs,
                                             length_for_normalization = length_for_normalization,
                                             seq_padding_idx = seq_padding_idx,
                                             t_array = t_array)
    loss, loss_fn_dict = out
    del out
    
    ### TODO: concat values from pred_layer_metrics, if there's any
    ###   so that code runs, just return empty dictionary for now
    pred_layer_metrics = dict()
    
    
    ### evaluate metrics
    metrics_dict = finalpred_instance.compile_metrics(true_out = true_out,
                                                      loss = loss,
                                                      loss_fn_dict = loss_fn_dict,
                                                      seq_padding_idx = seq_padding_idx)
    
    ##########################################
    ### COMPILE FINAL DICTIONARY TO RETURN   #
    ##########################################
    ### things that always get returned
    out_dict = {'loss': loss,
                'sum_neg_logP': loss_fn_dict['sum_neg_logP'],
                'neg_logP_length_normed': loss_fn_dict['neg_logP_length_normed']}
    
    # will have: perplexity_perSamp, ece
    # might also have: acc_perSamp, cm_perSamp
    out_dict = {**out_dict, **metrics_dict}
    
    
    ### optional things to add
    def write_optional_outputs(value_to_save, flag, varname_to_write):
        if flag:
            out_dict[varname_to_write] = value_to_save
    
    # intermediate values captured during training
    write_optional_outputs(value_to_save = embeddings_aux_dict['anc_layer_metrics'], 
                           flag = encoder_sow_outputs, 
                           varname_to_write = 'anc_layer_metrics')
    
    write_optional_outputs(value_to_save = embeddings_aux_dict['desc_layer_metrics'], 
                           flag = decoder_sow_outputs, 
                           varname_to_write = 'desc_layer_metrics')
    
    write_optional_outputs(value_to_save = pred_layer_metrics, 
                           flag = finalpred_sow_outputs, 
                           varname_to_write = 'pred_layer_metrics')
    
    # transformer attention weights
    if 'anc_attn_weights' in embeddings_aux_dict.keys():
        write_optional_outputs(value_to_save = embeddings_aux_dict['anc_attn_weights'], 
                               flag = True, 
                               varname_to_write = 'anc_attn_weights')
    
    if 'desc_attn_weights' in embeddings_aux_dict.keys():
        write_optional_outputs(value_to_save = embeddings_aux_dict['desc_attn_weights'], 
                               flag = True, 
                               varname_to_write = 'desc_attn_weights')
        
    # other arrays to return
    write_optional_outputs(value_to_save = anc_embeddings,
                           flag = return_anc_embs,
                           varname_to_write = 'final_ancestor_embeddings')
    
    write_optional_outputs(value_to_save = desc_embeddings,
                           flag = return_desc_embs,
                           varname_to_write = 'final_descendant_embeddings')
    
    if 'neg_logP_perSamp_perPos' in loss_fn_dict.keys():
        write_optional_outputs(value_to_save = loss_fn_dict['neg_logP_perSamp_perPos'],
                               flag = return_final_logprobs,
                               varname_to_write = 'final_logprobs')
    
    
    # instead of "final_logits," write whatever comes out of forward_pass_outputs
    if return_forward_pass_outputs:
        for varname_to_write, value_to_save in scan_fn_outputs[1].items():
            if varname_to_write.startswith('FPO_'):
                out_dict[varname_to_write] = value_to_save
    
    # always returned from out_dict:
    #     - loss; float
    #     - sum_neg_logP; (B,)
    #     - neg_logP_length_normed; (B,)
    #     - perplexity_perSamp; (B,)
    
    # returned, if using feedforward prediction head:
    #     - acc_perSamp; (B,)
    #     - cm_perSamp; (B, out_alph_size, out_alph_size)
        
    # returned if flag active:
    #     - anc_layer_metrics
    #     - desc_layer_metrics
    #     - pred_layer_metrics
    #     - anc_attn_weights 
    #     - desc_attn_weights 
    #     - final_ancestor_embeddings
    #     - final_descendant_embeddings
    #     - any outputs from forward_pass_outputs (like final_logits)
    #     - final_logprobs (i.e. AFTER log_softmax)
         
    
    # DEBUG: add extra values
    # out_dict = {**out_dict, **loss_fn_dict}
    return out_dict
