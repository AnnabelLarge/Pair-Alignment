#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 00:50:39 2023

@author: annabel_large

ABOUT:
======
train and eval functions for one batch of data


THIS USES THE WHOLE SEQUENCE LENGTH

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

from utils.sequence_length_helpers import selective_squeeze


###############################################################################
### TRAIN ON ONE BATCH    #####################################################
###############################################################################
def train_one_batch(batch, 
                    training_rngkey,
                    all_trainstates,  
                    max_seq_len,
                    max_align_len,
                    all_model_instances,
                    norm_loss_by,
                    interms_for_tboard, 
                    more_attributes: dict,
                    gap_tok = 43,
                    seq_padding_idx = 0,
                    align_idx_padding = -9):
    """
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
        > interms_for_tboard: decide whether or not to output intermediate 
                             histograms and scalars
        > norm_loss_by: what length to normalize losses by
        > more_attributes: extra params that are model specific (sometimes 
          need to be used by methods outside of call and setup....
          sometimes I hate flax.linen
    
    outputs:
        > metrics_outputs: dictionary of metrics and outputs                                  
        
    """
    #########################
    ### UNPACK FLAGS, FNS   #
    #########################
    # booleans for determining which sowed outputs to write to
    #   tensorboard
    encoder_sow_outputs = interms_for_tboard['encoder_sow_outputs']
    decoder_sow_outputs = interms_for_tboard['decoder_sow_outputs']
    finalpred_sow_outputs = interms_for_tboard['finalpred_sow_outputs']
    save_gradients = interms_for_tboard['gradients']
    save_updates = interms_for_tboard['optimizer']
    del interms_for_tboard
    
    
    ##################################
    ### UNPACK THE INPUTS, PREPROC   #
    ##################################
    concat_fn = more_attributes['concat_fn']
    
    ### unpack
    encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
    encoder_instance, decoder_instance, finalpred_instance = all_model_instances
    del all_model_instances, all_trainstates
    
    
    ### clip to max lengths, split into prefixes and suffixes
    batch_unaligned_seqs, batch_aligned_mats, t_array, _ = batch
    del batch
    
    # first clip
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :]
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    
    # split into prefixes and suffixes, to avoid confusion
    # prefixes: <s> A  B  C    the "a" in P(b | a, X, Y_{...j})
    #            |  |  |  |
    #            v  v  v  v
    # suffixes:  A  B  C <e>    the "b" in P(b | a, X, Y_{...j})
    aligned_mats_prefixes = clipped_aligned_mats[:,:-1,:]
    unaligned_seqs_prefixes = clipped_unaligned_seqs[:,:-1,:]
    aligned_mats_suffixes = clipped_aligned_mats[:,1:,:]
    del clipped_unaligned_seqs, clipped_aligned_mats
    
    
    ### unpack sequences
    # unaligned sequences used in __call__; final size is (B, max_seq_len)
    anc_seqs = unaligned_seqs_prefixes[...,0]
    desc_seqs = unaligned_seqs_prefixes[...,1]
    
    # precomputed alignment indices; final size is (B, max_align_len-1, 2)
    # don't include last token, since it's not used to predict any valid input
    align_idxes = aligned_mats_prefixes[...,-2:]
    
    # other unpacking/processing is model specific
    out = finalpred_instance.process_aligned_mats( prefixes = aligned_mats_prefixes,
                                                   suffixes = aligned_mats_suffixes,
                                                   norm_loss_by = norm_loss_by,
                                                   more_attributes = more_attributes,
                                                   seq_padding_idx = seq_padding_idx,
                                                   gap_tok = gap_tok )
    true_out, extra_features, length_for_normalization = out
    del out
    
    
    ### produce new keys for each network
    all_keys = jax.random.split(training_rngkey, num=4)
    training_rngkey, enc_key, dec_key, finalpred_key = all_keys
    del all_keys
    
    
    ############################################
    ### APPLY MODEL, EVALUATE LOSS AND GRADS   #
    ############################################
    def apply_model(encoder_params, decoder_params, finalpred_params):
        ### embed with ancestor encoder
        # anc_embeddings is (B, max_seq_len-1, H)
        out = encoder_instance.apply_seq_embedder_in_training(seqs = anc_seqs,
                                                              rng_key = enc_key,
                                                              params_for_apply = encoder_params,
                                                              seq_emb_trainstate = encoder_trainstate,
                                                              sow_outputs = encoder_sow_outputs)
        anc_embeddings, embeddings_aux_dict = out
        del out
        
        ### embed with descendant decoder
        # desc_embeddings is (B, max_seq_len-1, H)
        out = decoder_instance.apply_seq_embedder_in_training(seqs = desc_seqs,
                                                              rng_key = dec_key,
                                                              params_for_apply = decoder_params,
                                                              seq_emb_trainstate = decoder_trainstate,
                                                              sow_outputs = decoder_sow_outputs)
        desc_embeddings, to_add = out
        del out
        
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
        
        
        ### extract embeddings
        out = concat_fn(anc_encoded = anc_embeddings, 
                        desc_encoded = desc_embeddings,
                        extra_features = extra_features,
                        idx_lst = align_idxes,
                        seq_padding_idx = seq_padding_idx,
                        align_idx_padding = align_idx_padding)
        datamat_lst, alignment_padding_mask = out
        del out
        
        ### forward pass through prediction head
        mut = ['histograms','scalars'] if finalpred_sow_outputs else []
        out = finalpred_trainstate.apply_fn(variables = finalpred_params,
                                            datamat_lst = datamat_lst,
                                            t_array = t_array,
                                            padding_mask = alignment_padding_mask,
                                            training = True,
                                            sow_intermediates = finalpred_sow_outputs,
                                            mutable=mut,
                                            rngs={'dropout': finalpred_key})
        forward_pass_outputs, pred_sow_dict = out
        del out, mut
        
        pred_layer_metrics = {'histograms': pred_sow_dict.get( 'histograms', dict() ),
                              'scalars': pred_sow_dict.get( 'scalars', dict() )
                              }
        
        
        ### evaluate loglike in parts
        # this calculates sum of logprob over length of alignment
        out = finalpred_instance.neg_loglike_in_scan_fn(forward_pass_outputs = forward_pass_outputs,
                                                        true_out = true_out,
                                                        more_attributes = more_attributes,
                                                        seq_padding_idx = seq_padding_idx)
        sum_neg_logP_raw, intermeds_to_stack = out
        del out
        
        # some metrics need output from forward pass, so calculate those here
        to_add = finalpred_instance.compile_metrics_in_scan(forward_pass_outputs = forward_pass_outputs, 
                                                            seq_padding_idx = seq_padding_idx)
        intermeds_to_stack = {**intermeds_to_stack, **intermeds_to_stack}
        
        # output loss should be (T, B), even if no time provided
        if len(sum_neg_logP_raw.shape) == 1:
            sum_neg_logP_raw = sum_neg_logP_raw[None, :]
        
        # final loss calculation: normalize by desired alignment length, 
        #   possibly logsumexp across timepoints
        out = finalpred_instance.evaluate_loss_after_scan(scan_fn_outputs = (sum_neg_logP_raw, intermeds_to_stack),
                                                 length_for_normalization = length_for_normalization,
                                                 seq_padding_idx = seq_padding_idx,
                                                 t_array = t_array,
                                                 more_attributes = more_attributes)
        loss, aux_dict = out
        del intermeds_to_stack, out, sum_neg_logP_raw  
        
        ### return EVERYTHING
        aux_dict['embeddings_aux_dict'] = embeddings_aux_dict
        aux_dict['pred_layer_metrics'] = pred_layer_metrics
        
        for key, val in forward_pass_outputs.items():
            if key.startswith('FPO_'):
                aux_dict[key] = val
                
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
    #     - neg_logP_length_normed (B,): loss per sample
    #     - sum_neg_logP (B,): loss per sample NOT NORMALIZED BY LENGTH YET 
    #     - all forward pass outputs: logprob emit match, logprob emit ins,
    #         logprob transitions, and substitution rate matrics for tkf models
    
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
                'neg_logP_length_normed': aux_dict['neg_logP_length_normed'],
                'sum_neg_logP': aux_dict['sum_neg_logP'],
                'batch_loss': batch_loss,
                'batch_ave_acc': batch_ave_acc,
                'batch_ave_perpl': batch_ave_perpl
                }
    for key, val in aux_dict.items():
            if key.startswith('FPO_'):
                out_dict[key] = val
    
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
    #     - neg_logP_length_normed (B,); the loss per sample
    #     - all forward pass outputs: logprob emit match, logprob emit ins,
    #         logprob transitions, and substitution rate matrics for tkf models
    
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
                   max_seq_len,
                   max_align_len,
                   all_model_instances,  
                   norm_loss_by,
                   interms_for_tboard, 
                   more_attributes: dict,
                   gap_tok = 43,
                   seq_padding_idx = 0,
                   align_idx_padding = -9,
                   extra_args_for_eval: dict = dict(),
                   **kwargs):
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
    encoder_sow_outputs = interms_for_tboard['encoder_sow_outputs']
    decoder_sow_outputs = interms_for_tboard['decoder_sow_outputs']
    finalpred_sow_outputs = interms_for_tboard['finalpred_sow_outputs']
    
    # booleans for determining which intermediate arrays to return
    return_anc_embs = interms_for_tboard['ancestor_embeddings']
    return_desc_embs = interms_for_tboard['descendant_embeddings']
    return_forward_pass_outputs = interms_for_tboard['forward_pass_outputs']
    return_final_logprobs = interms_for_tboard['final_logprobs']
    del interms_for_tboard
    
    
    ##################################
    ### UNPACK THE INPUTS, PREPROC   #
    ##################################
    concat_fn = more_attributes['concat_fn']
    
    ### unpack
    encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
    encoder_instance, decoder_instance, finalpred_instance = all_model_instances
    del all_model_instances, all_trainstates
    
    
    ### clip to max lengths, split into prefixes and suffixes
    batch_unaligned_seqs, batch_aligned_mats, t_array, _ = batch
    del batch
    
    # first clip
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :]
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    
    # split into prefixes and suffixes, to avoid confusion
    # prefixes: <s> A  B  C    the "a" in P(b | a, X, Y_{...j})
    #            |  |  |  |
    #            v  v  v  v
    # suffixes:  A  B  C <e>    the "b" in P(b | a, X, Y_{...j})
    aligned_mats_prefixes = clipped_aligned_mats[:,:-1,:]
    unaligned_seqs_prefixes = clipped_unaligned_seqs[:,:-1,:]
    aligned_mats_suffixes = clipped_aligned_mats[:,1:,:]
    del clipped_unaligned_seqs, clipped_aligned_mats
    
    
    ### unpack sequences
    # unaligned sequences used in __call__; final size is (B, max_seq_len)
    anc_seqs = unaligned_seqs_prefixes[...,0]
    desc_seqs = unaligned_seqs_prefixes[...,1]
    
    # precomputed alignment indices; final size is (B, max_align_len-1, 2)
    # don't include last token, since it's not used to predict any valid input
    align_idxes = aligned_mats_prefixes[...,-2:]
    
    # other unpacking/processing is model specific
    out = finalpred_instance.process_aligned_mats( prefixes = aligned_mats_prefixes,
                                                   suffixes = aligned_mats_suffixes,
                                                   norm_loss_by = norm_loss_by,
                                                   more_attributes = more_attributes,
                                                   seq_padding_idx = seq_padding_idx,
                                                   gap_tok = gap_tok )
    true_out, extra_features, length_for_normalization = out
    del out

    
    #######################
    ### Apply the model   #
    #######################
    ### embed with ancestor encoder
    # anc_embeddings is (B, max_seq_len-1, H)
    out = encoder_instance.apply_seq_embedder_in_eval(seqs = anc_seqs,
                                                      final_trainstate = encoder_trainstate,
                                                      sow_outputs = encoder_sow_outputs,
                                                      extra_args_for_eval = extra_args_for_eval)
    anc_embeddings, embeddings_aux_dict = out
    del out
    
    ### embed with descendant decoder
    # desc_embeddings is (B, max_seq_len-1, H)
    out = decoder_instance.apply_seq_embedder_in_eval(seqs = desc_seqs,
                                                      final_trainstate = decoder_trainstate,
                                                      sow_outputs = decoder_sow_outputs,
                                                      extra_args_for_eval = extra_args_for_eval)
    desc_embeddings, to_add = out
    del out
    
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
    
    ### extract embeddings
    out = concat_fn(anc_encoded = anc_embeddings, 
                    desc_encoded = desc_embeddings,
                    extra_features = extra_features,
                    idx_lst = align_idxes,
                    seq_padding_idx = seq_padding_idx,
                    align_idx_padding = align_idx_padding)
    datamat_lst, alignment_padding_mask = out
    del out
    
        
    ### forward pass through prediction head
    mut = ['histograms','scalars'] if finalpred_sow_outputs else []
    out = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                         datamat_lst = datamat_lst,
                                         t_array = t_array,
                                         padding_mask = alignment_padding_mask,
                                         training = False,
                                         sow_intermediates = finalpred_sow_outputs,
                                         mutable=mut
                                         )
    forward_pass_outputs, pred_sow_dict = out
    del mut, out
        
    pred_layer_metrics = {'histograms': pred_sow_dict.get( 'histograms', dict() ),
                          'scalars': pred_sow_dict.get( 'scalars', dict() )
                          }
    
    
    ### evaluate loglike in parts
    # this calculates sum of logprob over length of alignment
    out = finalpred_instance.neg_loglike_in_scan_fn(forward_pass_outputs = forward_pass_outputs,
                                                   true_out = true_out,
                                                   more_attributes = more_attributes,
                                                   seq_padding_idx = seq_padding_idx)
    sum_neg_logP_raw, intermeds_to_stack = out
    del out
        
    # some metrics need output from forward pass, so calculate those here
    to_add = finalpred_instance.compile_metrics_in_scan(forward_pass_outputs = forward_pass_outputs, 
                                                        seq_padding_idx = seq_padding_idx)
    intermeds_to_stack = {**intermeds_to_stack, **to_add}
    
    # output loss should be (T, B), even if no time provided
    if len(sum_neg_logP_raw.shape) == 1:
        sum_neg_logP_raw = sum_neg_logP_raw[None, :]
    
    # possibly return forward pass outputs
    if return_forward_pass_outputs:
        for key in forward_pass_outputs:
            if key.startswith('FPO_'):
                intermeds_to_stack[key] = forward_pass_outputs[key] 
        
    
    ### final loss calculation: normalize by desired alignment length, 
    ###   possibly logsumexp across timepoints
    out = finalpred_instance.evaluate_loss_after_scan(scan_fn_outputs = (sum_neg_logP_raw, intermeds_to_stack),
                                             length_for_normalization = length_for_normalization,
                                             seq_padding_idx = seq_padding_idx,
                                             t_array = t_array,
                                             more_attributes = more_attributes)
    loss, loss_fn_dict = out
    del out, sum_neg_logP_raw  
    
    
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
        for varname_to_write, value_to_save in intermed_dict.items():
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
