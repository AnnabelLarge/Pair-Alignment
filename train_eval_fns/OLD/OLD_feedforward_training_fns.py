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
from models.sequence_embedders.concatenation_fns import extract_embs


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
                    add_prev_alignment_info,
                    gap_idx = 43,
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
        > add_prev_alignment_info: add previous alignment label? 
                                   makes this pairHMM like
    
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
    
    
    ####################
    ### UNPACK, CLIP   #
    ####################
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
    
    
    ### produce new keys for each network
    all_keys = jax.random.split(training_rngkey, num=4)
    training_rngkey, enc_key, dec_key, finalpred_key = all_keys
    del all_keys
    
    
    ##################
    ### PREPROCESS   #
    ##################
    ### unpack features
    # unaligned sequences used in __call__; final size is (B, max_seq_len)
    anc_seqs = clipped_unaligned_seqs[...,0]
    desc_seqs = clipped_unaligned_seqs[...,1]
    
    # split into prefixes and suffixes, to avoid confusion
    # prefixes: <s> A  B  C    the "a" in P(b | a, X, Y_{...j})
    #            |  |  |  |
    #            v  v  v  v
    # suffixes:  A  B  C <e>    the "b" in P(b | a, X, Y_{...j})
    aligned_mats_prefixes = clipped_aligned_mats[:,:-1,:]
    unaligned_seqs_prefixes = clipped_unaligned_seqs[:,:-1,:]
    aligned_mats_suffixes = clipped_aligned_mats[:,1:,:]
    del clipped_unaligned_seqs, clipped_aligned_mats
    
    # precomputed alignment indices; final size is (B, max_align_len-1, 2)
    # don't include last token, since it's not used to predict any valid input
    align_idxes = aligned_mats_prefixes[...,-2:]
    
    
    ### true_out
    # only need the alignment-augmented descendant; dim0=0
    true_out = aligned_mats_suffixes[...,0]
    
    
    ### optionally, one-hot encode the previous alignment state as an 
    ### extra input feature (found at dim0=1)
    if add_prev_alignment_info:
        extra_features = activation.one_hot( x = aligned_mats_prefixes[...,1], 
                                             num_classes = 6 )
    else:
        extra_features = None
        
        
    ### length_for_normalization
    # don't include pad or eos
    length_for_normalization = jnp.where( ~jnp.isin(true_out, jnp.array([seq_padding_idx, 1]) ), 
                                          True, 
                                          False ).sum(axis=1)
    
    if norm_loss_by == 'desc_len':
        num_gaps = jnp.where( true_out == (gap_idx-1), 
                              True, 
                              False ).sum(axis=1)
        length_for_normalization = length_for_normalization - num_gaps
    
    
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
        out = extract_embs(anc_encoded = anc_embeddings, 
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
                                            padding_mask = alignment_padding_mask,
                                            training = True,
                                            sow_intermediates = finalpred_sow_outputs,
                                            mutable=mut,
                                            rngs={'dropout': finalpred_key})
        final_logits, pred_sow_dict = out
        del out, mut
        
        pred_layer_metrics = {'histograms': pred_sow_dict.get( 'histograms', dict() ),
                              'scalars': pred_sow_dict.get( 'scalars', dict() )
                              }
        
        
        ### evaluate loglike
        out = finalpred_instance.apply_ce_loss( final_logits = final_logits,
                                                true_out = true_out,
                                                length_for_normalization = length_for_normalization,
                                                seq_padding_idx = seq_padding_idx )        
        loss, loss_intermeds = out
        del out
        
        # create aux dictionary
        aux_dict = {'sum_neg_logP': loss_intermeds['sum_neg_logP'],
                    'neg_logP_length_normed': loss_intermeds['neg_logP_length_normed'],
                    'final_logits': final_logits,
                    'embeddings_aux_dict': embeddings_aux_dict,
                    'pred_layer_metrics': pred_layer_metrics}
        return (loss, aux_dict)
    
    
    ### set up the grad functions, based on above loss function
    grad_fn = jax.value_and_grad(apply_model, argnums=[0,1,2], has_aux=True)
    
    (batch_loss, aux_dict), all_grads = grad_fn(encoder_trainstate.params, 
                                                decoder_trainstate.params, 
                                                finalpred_trainstate.params)
    
    enc_gradient, dec_gradient, finalpred_gradient = all_grads
    del all_grads
    
    # aux_dict contains:
    #   - final_logits (B, L)
    #   - neg_logP_length_normed (B,): loss per sample
    #   - sum_neg_logP (B,): loss per sample NOT NORMALIZED BY LENGTH YET 
    #   - sowed dictionaries
    
    
    # get other metrics
    #   normally, because the gap token is the last element in the alphabet,
    #   out_alph_size_with_pad = gap_idx + 1, but here, I've removed <bos>,
    #   which was encoded as 1. Entire alphabet shifted down, so now 
    #   out_alph_size_with_pad = gap_idx
    metrics = finalpred_instance.compile_metrics( true_out = true_out,     
                                                  final_logits = aux_dict['final_logits'],
                                                  loss = batch_loss,
                                                  neg_logP_length_normed = aux_dict['neg_logP_length_normed'],
                                                  seq_padding_idx = 0,
                                                  out_alph_size_with_pad = gap_idx )
    
    batch_ave_perpl = jnp.mean(metrics['perplexity_perSamp'])
    batch_ave_acc = jnp.mean(metrics['acc_perSamp'])
        
    
    ###########################
    ### RECORD UPDATES MADE   #
    ###########################
    if update_grads:
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
    
    elif not update_grads:
        new_encoder_trainstate = encoder_trainstate
        new_decoder_trainstate = decoder_trainstate
        new_finalpred_trainstate = finalpred_trainstate
    
    
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
    #     - batch_ave_acc; float 
    #     - sum_neg_logP (B,); the loss per sample BEFORE normalizing by length
    #     - neg_logP_length_normed (B,); the loss per sample
    
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
                   add_prev_alignment_info,
                   gap_idx = 43,
                   seq_padding_idx = 0,
                   align_idx_padding = -9,
                   extra_args_for_eval: dict = dict() ):
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
    
    
    ####################
    ### UNPACK, CLIP   #
    ####################
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
    
    
    ##################
    ### PREPROCESS   #
    ##################
    ### unpack features
    # unaligned sequences used in __call__; final size is (B, max_seq_len)
    anc_seqs = clipped_unaligned_seqs[...,0]
    desc_seqs = clipped_unaligned_seqs[...,1]
    
    # split into prefixes and suffixes, to avoid confusion
    # prefixes: <s> A  B  C    the "a" in P(b | a, X, Y_{...j})
    #            |  |  |  |
    #            v  v  v  v
    # suffixes:  A  B  C <e>    the "b" in P(b | a, X, Y_{...j})
    aligned_mats_prefixes = clipped_aligned_mats[:,:-1,:]
    unaligned_seqs_prefixes = clipped_unaligned_seqs[:,:-1,:]
    aligned_mats_suffixes = clipped_aligned_mats[:,1:,:]
    del clipped_unaligned_seqs, clipped_aligned_mats
    
    # precomputed alignment indices; final size is (B, max_align_len-1, 2)
    # don't include last token, since it's not used to predict any valid input
    align_idxes = aligned_mats_prefixes[...,-2:]
    
    
    ### true_out
    # only need the alignment-augmented descendant; dim0=0
    true_out = aligned_mats_suffixes[...,0]
    
    
    ### optionally, one-hot encode the alignment state as an 
    ### extra input feature (found at dim0=1)
    if add_prev_alignment_info:
        extra_features = activation.one_hot( x = aligned_mats_prefixes[...,1], 
                                             num_classes = 6 )
    else:
        extra_features = None
        
        
    ### length_for_normalization
    # don't include pad or eos
    length_for_normalization = jnp.where( ~jnp.isin(true_out, jnp.array([seq_padding_idx, 1]) ), 
                                          True, 
                                          False ).sum(axis=1)
                                         
    if norm_loss_by == 'desc_len':
        num_gaps = jnp.where( true_out == (gap_idx-1), 
                              True, 
                              False ).sum(axis=1)
        length_for_normalization = length_for_normalization - num_gaps
        
    
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
    out = extract_embs(anc_encoded = anc_embeddings, 
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
                                         padding_mask = alignment_padding_mask,
                                         training = False,
                                         sow_intermediates = finalpred_sow_outputs,
                                         mutable=mut
                                         )
    final_logits, pred_sow_dict = out
    del mut, out
        
    pred_layer_metrics = {'histograms': pred_sow_dict.get( 'histograms', dict() ),
                          'scalars': pred_sow_dict.get( 'scalars', dict() )
                          }
    
    
    ### evaluate loglike
    out = finalpred_instance.apply_ce_loss( final_logits = final_logits,
                                            true_out = true_out,
                                            length_for_normalization = length_for_normalization,
                                            seq_padding_idx = seq_padding_idx )        
    loss, loss_intermeds = out
    del out
    
    # get other metrics
    #   normally, because the gap token is the last element in the alphabet,
    #   out_alph_size_with_pad = gap_idx + 1, but here, I've removed <bos>,
    #   which was encoded as 1. Entire alphabet shifted down, so now 
    #   out_alph_size_with_pad = gap_idx
    metrics = finalpred_instance.compile_metrics( true_out = true_out,     
                                                  final_logits = final_logits,
                                                  loss = loss,
                                                  neg_logP_length_normed = loss_intermeds['neg_logP_length_normed'],
                                                  seq_padding_idx = 0,
                                                  out_alph_size_with_pad = gap_idx )
    
    ##########################################
    ### COMPILE FINAL DICTIONARY TO RETURN   #
    ##########################################
    ### things that always get returned
    out_dict = {'batch_loss': loss,
                'sum_neg_logP': loss_intermeds['sum_neg_logP'],
                'neg_logP_length_normed': loss_intermeds['neg_logP_length_normed']}
    
    # will have: perplexity_perSamp, ece, acc_perSamp, cm_perSamp
    out_dict = {**out_dict, **metrics}
    
    
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
    
    write_optional_outputs(value_to_save = loss_intermeds['neg_logP_perSamp_perPos'],
                           flag = return_final_logprobs,
                           varname_to_write = 'final_logprobs')

    # always returned from out_dict:
    #     - loss; float
    #     - sum_neg_logP; (B,)
    #     - neg_logP_length_normed; (B,)
    #     - perplexity_perSamp; (B,)
    #     - acc_perSamp; (B,)
    #     - cm_perSamp; (B, out_alph_size-1, out_alph_size-1)
    #     - ece; float
        
    # returned if flag active:
    #     - anc_layer_metrics
    #     - desc_layer_metrics
    #     - pred_layer_metrics
    #     - anc_attn_weights 
    #     - desc_attn_weights 
    #     - final_ancestor_embeddings
    #     - final_descendant_embeddings
    #     - final_logprobs (i.e. AFTER log_softmax)
         
    return out_dict
