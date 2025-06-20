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
from jax import Array
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
                    encoder_update_fn,
                    max_seq_len,
                    max_align_len,
                    interms_for_tboard, 
                    t_array,  
                    concat_fn, 
                    norm_loss_by_for_reporting: str='desc_len',
                    update_grads: bool = True,
                    gap_tok: int = 43,
                    seq_padding_idx: int = 0,
                    align_idx_padding: int = -9,
                    *args,
                    **kwargs):
    """
    Jit-able function to apply the model to one batch of samples, evaluate loss
    and collect gradients, then update model parameters
    
    regular inputs:
        > batch: batch from a pytorch dataloader
        > training_rngkey: the rng key
        > all_trainstates: the models + parameters
    
    static inputs:
        > encoder_update_fn: updates the encoder trainstate; defined by 
          encoder_instance.update_seq_embedder_tstate
        > max_seq_len: max length of unaligned seqs matrix (used to control 
                       number of jit-compiled versions of this function)
        > max_align_len: max length of alignment matrix (used to control 
                         number of jit-compiled versions of this function) 
        > norm_loss_by_for_reporting: when reporting loss, normalize by some
                    sequence length; default is desc_len
        > interms_for_tboard: decide whether or not to output intermediate 
                             histograms and scalars
        > update_grads: only turn off when debugging
        > concat_fn: what function to use to concatenate embedded seq inputs

    static inputs, specific to neural hmm:
        > t_array: one time array for all samples (T,), or None
    
    outputs:
        > metrics_outputs: dictionary of metrics and outputs                                  
        
    """
    # which times to use 
    if t_array is None:
        times_for_matrices = batch[2] #(B,)
    
    else:
        times_for_matrices = t_array #(T,)
        
        
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
    
    
    ##############################
    ### UNPACK, CLIP, MAKE RNGS  #
    ##############################
    # unpack
    encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
    del all_trainstates
    
    # clip to max lengths, split into prefixes and suffixes
    batch_unaligned_seqs = batch[0] #(B, L, 2)
    batch_aligned_mats = batch[1] #(B, L, 5)
    del batch
    
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :] #(B, L_seq, 2)
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :] #(B, L_align, 5)
    
    # produce new keys for each network
    all_keys = jax.random.split(training_rngkey, num=4)
    training_rngkey, enc_key, dec_key, finalpred_key = all_keys
    del all_keys
    
    
    ##########################
    ### PREPARE THE INPUTS   #
    ##########################
    ### unpack features
    # unaligned sequences used in __call__
    anc_seqs = clipped_unaligned_seqs[...,0] # (B, L_seq)
    desc_seqs = clipped_unaligned_seqs[...,1] # (B, L_seq)
    
    # split into prefixes and suffixes, to avoid confusion
    # prefixes: <s> A  B  C    the "a" in P(b | a, X, Y_{...j})
    #            |  |  |  |
    #            v  v  v  v
    # suffixes:  A  B  C <e>    the "b" in P(b | a, X, Y_{...j})
    aligned_mats_prefixes = clipped_aligned_mats[:,:-1,:]  # (B, L_align-1, 5)
    aligned_mats_suffixes = clipped_aligned_mats[:,1:,:] # (B, L_align-1, 5)
    del clipped_unaligned_seqs, clipped_aligned_mats
    
    # precomputed alignment indices
    # don't include last token, since it's not used to predict any valid input
    align_idxes = aligned_mats_prefixes[...,-2:] # (B, L_align-1, 2)
    
    
    ### prepare true_out
    # need three things: gapped ancestor, gapped descendant, and 
    # state transitions (a -> b transitions)
    gapped_anc_desc = aligned_mats_suffixes[...,:2] #(B, L_align-1, 2)
    from_states = aligned_mats_prefixes[...,2][..., None] #(B, L_align-1, 1)
    to_states = aligned_mats_suffixes[...,2][..., None] #(B, L_align-1, 1)
    true_out = jnp.concatenate( [ gapped_anc_desc,
                                  from_states,
                                  to_states ],
                                axis = -1 ) #(B, L_align-1, 4)
    
    # when reporting, normalize the loss by a length (but this is NOT the 
    #   objective function)
    length_for_normalization_for_reporting = true_out[...,1] != seq_padding_idx.sum(axis=1)
    
    if norm_loss_by_for_reporting == 'desc_len':
        num_gaps = (true_out[...,1] == gap_tok).sum(axis=1)
        length_for_normalization_for_reporting = length_for_normalization_for_reporting - num_gaps
        
        
    ############################################
    ### APPLY MODEL, EVALUATE LOSS AND GRADS   #
    ############################################
    def apply_model(encoder_params, 
                    decoder_params, 
                    finalpred_params, 
                    t_for_training):
        ### embed with ancestor encoder
        # anc_embeddings is (B, L_seq-1, H)
        out = encoder_trainstate.apply_fn( variables = encoder_params,
                                           seqs = anc_seqs,
                                           rng_key = enc_key,
                                           params_for_apply = encoder_params,
                                           seq_emb_trainstate = encoder_trainstate,
                                           sow_outputs = encoder_sow_outputs,
                                           method = 'apply_seq_embedder_in_training' )
        
        anc_embeddings, embeddings_aux_dict = out
        del out
        
        
        ### embed with descendant decoder
        # desc_embeddings is (B, L_seq-1, H)
        out = decoder_trainstate.apply_fn( variables = decoder_params,
                                        seqs = desc_seqs,
                                        rng_key = dec_key,
                                        params_for_apply = decoder_params,
                                        seq_emb_trainstate = decoder_trainstate,
                                        sow_outputs = decoder_sow_outputs,
                                        method = 'apply_seq_embedder_in_training' )
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
                        idx_lst = align_idxes,
                        seq_padding_idx = seq_padding_idx,
                        align_idx_padding = align_idx_padding)
        datamat_lst, padding_mask = out
        del out
        
        # add previous alignment state
        datamat_lst.append( from_states ) #(B, L_align-1, 1)
        
        ### forward pass through prediction head, to get scoring matrices
        mut = ['histograms','scalars'] if finalpred_sow_outputs else []
        out = finalpred_trainstate.apply_fn(variables = finalpred_params,
                                            datamat_lst = datamat_lst,
                                            t_array = t_for_training,
                                            padding_mask = padding_mask,
                                            training = True,
                                            sow_intermediates = finalpred_sow_outputs,
                                            mutable=mut,
                                            rngs={'dropout': finalpred_key})
        
        # forward_pass_scoring_matrices has the keys:
        # logprob_emit_match: (T,B,L,A,A) or (B,L,A,A)
        # logprob_emit_indel: (B,L,A)
        # logprob_transits: (T,B,L,S,S) or (B,L,S,S)
        # corr: a tuple of two arrays; each either (T,B) or (B,)
        # approx_flags_dict: a dictionary of things (see model code)
        # subs_model_params: a dictionary of things (see model code)
        # indel_model_params: a dictionary of things (see model code)
        forward_pass_scoring_matrices, pred_sow_dict = out
        del out, mut
        
        pred_layer_metrics = {'histograms': pred_sow_dict.get( 'histograms', dict() ),
                              'scalars': pred_sow_dict.get( 'scalars', dict() )
                              }
        
        
        ### evaluate loglike of true alignments
        # this calculates sum of logprob over length of alignment
        logprob_perSamp_perTime = finalpred_trainstate.apply_fn( variables = finalpred_params,
                                                         logprob_emit_match = forward_pass_scoring_matrices['logprob_emit_match'], 
                                                         logprob_emit_indel = forward_pass_scoring_matrices['logprob_emit_indel'],
                                                         logprob_transits = forward_pass_scoring_matrices['logprob_transits'],
                                                         corr = forward_pass_scoring_matrices['corr'],
                                                         true_out = true_out,
                                                         padding_idx = seq_padding_idx,
                                                         method = 'neg_loglike_in_scan_fn') #(T,B) or (B,)
        
        # final loss calculation; possibly logsumexp across timepoints
        out = finalpred_trainstate.apply_fn( variables = finalpred_params,
                                             logprob_perSamp_perTime = logprob_perSamp_perTime,
                                             length_for_normalization_for_reporting = length_for_normalization_for_reporting,
                                             t_array = t_array,
                                             padding_idx = seq_padding_idx,
                                             method = 'evaluate_loss_after_scan' )
        loss, aux_dict = out
        del out, logprob_perSamp_perTime  
        
        
        ### return EVERYTHING
        aux_dict['embeddings_aux_dict'] = embeddings_aux_dict
        aux_dict['pred_layer_metrics'] = pred_layer_metrics
        
        for key, val in forward_pass_scoring_matrices.items():
            aux_dict[f'scormat_{key}'] = val
        
        return (loss, aux_dict)
    
    
    ### set up the grad functions, based on above loss function
    grad_fn = jax.value_and_grad(apply_model, 
                                 argnums=[0,1,2], 
                                 has_aux=True)
    
    (batch_loss, aux_dict), all_grads = grad_fn(encoder_trainstate.params, 
                                                decoder_trainstate.params, 
                                                finalpred_trainstate.params,
                                                t_array = times_for_matrices)
    
    enc_gradient, dec_gradient, finalpred_gradient = all_grads
    del all_grads
    
    
    ### evaluate metrics BEFORE updating parameters
    batch_ave_perpl = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                                     loss_fn_dict = aux_dict,
                                                     method = 'get_perplexity_per_sample' ).mean()
    
    ece = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                         loss = batch_loss,
                                         method = 'get_ece' )
    
    
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
        # encoder_update_fn = encoder_instance.update_seq_embedder_tstate
        new_encoder_trainstate = encoder_update_fn(tstate = encoder_trainstate,
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
                'sum_neg_logP': aux_dict['sum_neg_logP'],
                'neg_logP_length_normed': aux_dict['neg_logP_length_normed'],
                'batch_loss': batch_loss,
                'batch_ave_perpl': batch_ave_perpl
                }
    for key, val in aux_dict.items():
        if key.startswith('scormat_'):
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
    if update_grads:
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
    #     - sum_neg_logP (B,); the loss per sample
    #     - neg_logP_length_normed (B,); the loss per sample, normalized by seq length
    #     - all scoring matrices from forward pass
    
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
                    interms_for_tboard, 
                    t_array,  
                    concat_fn, 
                    norm_loss_by_for_reporting: str='desc_len',
                    update_grads: bool = True,
                    gap_tok: int = 43,
                    seq_padding_idx: int = 0,
                    align_idx_padding: int = -9,
                    extra_args_for_eval: dict = dict(),
                    *args,
                    **kwargs):
    """
    Jit-able function to apply the model to one batch of samples, evaluate loss
    
    regular inputs:
        > batch: batch from a pytorch dataloader
        > all_trainstates: the models + parameters
    
    static inputs:
        > max_seq_len: max length of unaligned seqs matrix (used to control 
                       number of jit-compiled versions of this function)
        > max_align_len: max length of alignment matrix (used to control 
                         number of jit-compiled versions of this function) 
        > norm_loss_by_for_reporting: when reporting loss, normalize by some
                    sequence length; default is desc_len
        > interms_for_tboard: decide whether or not to output intermediate 
                             histograms and scalars
        > update_grads: only turn off when debugging
        > concat_fn: what function to use to concatenate embedded seq inputs
        > extra_args_for_eval: extra inputs for custom eval functions

    static inputs, specific to neural hmm:
        > t_array: one time array for all samples (T,), or None
    
    outputs:
        > metrics_outputs: dictionary of metrics and outputs 
    """
    # which times to use 
    if t_array is None:
        times_for_matrices = batch[2] #(B,)
    
    else:
        times_for_matrices = t_array #(T,)
    
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
    del interms_for_tboard
    
    
    ####################
    ### UNPACK, CLIP   #
    ####################
    # unpack
    encoder_trainstate, decoder_trainstate, finalpred_trainstate = all_trainstates
    del all_trainstates
    
    # clip to max lengths, split into prefixes and suffixes
    batch_unaligned_seqs = batch[0] #(B, L, 2)
    batch_aligned_mats = batch[1] #(B, L, 5)
    del batch
    
    clipped_unaligned_seqs = batch_unaligned_seqs[:, :max_seq_len, :] #(B, L_seq, 2)
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :] #(B, L_align, 5)
    
    
    ##########################
    ### PREPARE THE INPUTS   #
    ##########################
    ### unpack features
    # unaligned sequences used in __call__
    anc_seqs = clipped_unaligned_seqs[...,0] # (B, L_seq)
    desc_seqs = clipped_unaligned_seqs[...,1] # (B, L_seq)
    
    # split into prefixes and suffixes, to avoid confusion
    # prefixes: <s> A  B  C    the "a" in P(b | a, X, Y_{...j})
    #            |  |  |  |
    #            v  v  v  v
    # suffixes:  A  B  C <e>    the "b" in P(b | a, X, Y_{...j})
    aligned_mats_prefixes = clipped_aligned_mats[:,:-1,:] # (B, L_align-1, 5)
    aligned_mats_suffixes = clipped_aligned_mats[:,1:,:] # (B, L_align-1, 5)
    del clipped_unaligned_seqs, clipped_aligned_mats
    
    # precomputed alignment indices; final size is (B, max_align_len-1, 2)
    # don't include last token, since it's not used to predict any valid input
    align_idxes = aligned_mats_prefixes[...,-2:]
    
    
    ### prepare true_out
    # need three things: gapped ancestor, gapped descendant, and 
    # state transitions (a -> b transitions)
    gapped_anc_desc = aligned_mats_suffixes[...,:2] #(B, L_align-1, 2)
    from_states = aligned_mats_prefixes[...,2][..., None] #(B, L_align-1, 1)
    to_states = aligned_mats_suffixes[...,2][..., None] #(B, L_align-1, 1)
    true_out = jnp.concatenate( [ gapped_anc_desc,
                                  from_states,
                                  to_states ],
                                axis = -1 ) #(B, L_align-1, 4)
    
    
    # when reporting, normalize the loss by a length (but this is NOT the 
    #   objective function)
    length_for_normalization_for_reporting = true_out[...,1] != seq_padding_idx.sum(axis=1)
    
    if norm_loss_by_for_reporting == 'desc_len':
        num_gaps = (true_out[...,1] == gap_tok).sum(axis=1)
        length_for_normalization_for_reporting = length_for_normalization_for_reporting - num_gaps
    

    
    #######################
    ### Apply the model   #
    #######################
    ### embed with ancestor encoder
    # anc_embeddings is (B, L_seq-1, H)
    out = encoder_trainstate.apply_fn( variables = encoder_trainstate.params,
                                       seqs = anc_seqs,
                                       rng_key = enc_key,
                                       params_for_apply = encoder_params,
                                       seq_emb_trainstate = encoder_trainstate,
                                       sow_outputs = encoder_sow_outputs,
                                       method = 'apply_seq_embedder_in_eval' )
    
    anc_embeddings, embeddings_aux_dict = out
    del out
    
    
    ### embed with descendant decoder
    # desc_embeddings is (B, L_seq-1, H)
    out = decoder_trainstate.apply_fn( variables = decoder_trainstate.params,
                                    seqs = desc_seqs,
                                    rng_key = dec_key,
                                    params_for_apply = decoder_params,
                                    seq_emb_trainstate = decoder_trainstate,
                                    sow_outputs = decoder_sow_outputs,
                                    method = 'apply_seq_embedder_in_eval' )
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
                    idx_lst = align_idxes,
                    seq_padding_idx = seq_padding_idx,
                    align_idx_padding = align_idx_padding)
    datamat_lst, alignment_padding_mask = out
    del out
    
    # add previous alignment state
    datamat_lst.append( from_states ) #(B, L_align-1, 1)
    
        
    ### forward pass through prediction head, to get scoring matrices
    mut = ['histograms','scalars'] if finalpred_sow_outputs else []
    out = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                         datamat_lst = datamat_lst,
                                         t_array = times_for_matrices,
                                         padding_mask = alignment_padding_mask,
                                         training = False,
                                         sow_intermediates = finalpred_sow_outputs,
                                         mutable=mut
                                         )
    # forward_pass_scoring_matrices has the keys:
    # logprob_emit_match: (T,B,L,A,A) or (B,L,A,A)
    # logprob_emit_indel: (B,L,A)
    # logprob_transits: (T,B,L,S,S) or (B,L,S,S)
    # corr: a tuple of two arrays; each either (T,B) or (B,)
    # approx_flags_dict: a dictionary of things (see model code)
    # subs_model_params: a dictionary of things (see model code)
    # indel_model_params: a dictionary of things (see model code)
    forward_pass_scoring_matrices, pred_sow_dict = out
    del mut, out
        
    pred_layer_metrics = {'histograms': pred_sow_dict.get( 'histograms', dict() ),
                          'scalars': pred_sow_dict.get( 'scalars', dict() )
                          }
    
    
    ### evaluate loglike of true alignments
    # this calculates sum of logprob over length of alignment
    logprob_perSamp_perTime = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                                     logprob_emit_match = forward_pass_scoring_matrices['logprob_emit_match'], 
                                                     logprob_emit_indel = forward_pass_scoring_matrices['logprob_emit_indel'],
                                                     logprob_transits = forward_pass_scoring_matrices['logprob_transits'],
                                                     corr = forward_pass_scoring_matrices['corr'],
                                                     true_out = true_out,
                                                     padding_idx = seq_padding_idx,
                                                     method = 'neg_loglike_in_scan_fn') #(T,B) or (B,)
     
    # final loss calculation; possibly logsumexp across timepoints
    out = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                         logprob_perSamp_perTime = logprob_perSamp_perTime,
                                         length_for_normalization_for_reporting = length_for_normalization_for_reporting,
                                         t_array = t_array,
                                         padding_idx = seq_padding_idx,
                                         method = 'evaluate_loss_after_scan' )
    loss, loss_fn_dict = out
    del out, logprob_perSamp_perTime  
    
    
    ### evaluate metrics
    perplexity_perSamp = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                                     loss_fn_dict = loss_fn_dict,
                                                     method = 'get_perplexity_per_sample' )
    
    ece = finalpred_trainstate.apply_fn( variables = finalpred_trainstate.params,
                                         loss = loss,
                                         method = 'get_ece' )
    
    
    ##########################################
    ### COMPILE FINAL DICTIONARY TO RETURN   #
    ##########################################
    ### things that always get returned
    out_dict = {'batch_loss': loss,
                'sum_neg_logP': loss_fn_dict['sum_neg_logP'],
                'neg_logP_length_normed': loss_fn_dict['neg_logP_length_normed'],
                'perplexity_perSamp': perplexity_perSamp,
                'ece': ece}
    
    
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
    
    # instead of "final_logits," write whatever comes out of forward_pass_outputs
    if return_forward_pass_outputs:
        for key, val in aux_dict.items():
            if key.startswith('scormat_'):
                out_dict[key] = val
    
    # always returned from out_dict:
    #     - loss; float
    #     - sum_neg_logP; (B,)
    #     - neg_logP_length_normed; (B,)
    #     - perplexity_perSamp; (B,)
    
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
