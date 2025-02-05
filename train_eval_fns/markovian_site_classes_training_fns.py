#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:50:04 2025

@author: annabel
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


def train_one_batch(batch, 
                    training_rngkey,
                    pairhmm_trainstate,  
                    max_align_len,
                    interms_for_tboard
                    **kwargs):
    
    finalpred_sow_outputs = interms_for_tboard['finalpred_sow_outputs']
    
    _, batch_aligned_mats, t_array, _ = batch
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    del batch
    
    def apply_model(pairhmm_params):
        out = pairhmm_trainstate.apply_fn(variables = pairhmm_params,
                                          aligned_inputs = clipped_aligned_mats,
                                          t_array = t_array,
                                          sow_outputs = finalpred_sow_outputs)
        (loss, aux_dict), pred_sow_dict = out
        aux_dict = {**aux_dict, **pred_sow_dict}
        
        return loss, aux_dict
    
    grad_fn = jax.value_and_grad(apply_model, has_aux=True)
    (batch_loss, aux_dict), grad = grad_fn
 
    # update gradients 
    updates, new_opt_state = pairhmm_trainstate.tx.update(grad)
    new_params = optax.apply_updates(pairhmm_trainstate.params,
                                     updates)
    new_trainstate = pairhmm_trainstate.replace(params = new_params,
                                                opt_state = new_opt_state)
    
    
    ### other metrics
    # perplexity per sample
    neg_logP_length_normed = aux_dict['neg_logP_length_normed']
    perplexity_perSamp = jnp.exp(neg_logP_length_normed) #(B,)
    
    # exponentiated cross entropy
    ece = jnp.exp(loss)
    
    out_dict = {'neg_logP_length_normed': aux_dict['neg_logP_length_normed'],
                'sum_neg_logP': aux_dict['sum_neg_logP'],
                'ece': ece,
                'batch_loss': batch_loss,
                'batch_ave_perpl': jnp.mean(perplexity_perSamp) }
    
    return out_dict, new_trainstate


def eval_one_batch(batch, 
                    pairhmm_trainstate,  
                    max_align_len,
                    interms_for_tboard
                    **kwargs):
    finalpred_sow_outputs = interms_for_tboard['finalpred_sow_outputs']
    
    _, batch_aligned_mats, t_array, _ = batch
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    del batch
    
    out = pairhmm_trainstate.apply_fn(variables = pairhmm_trainstate.params,
                                      aligned_inputs = clipped_aligned_mats,
                                      t_array = t_array,
                                      sow_outputs = finalpred_sow_outputs)
    (loss, aux_dict), pred_sow_dict = out
    aux_dict = {**aux_dict, **pred_sow_dict}
    
    ### other metrics
    # perplexity per sample
    neg_logP_length_normed = aux_dict['neg_logP_length_normed']
    perplexity_perSamp = jnp.exp(neg_logP_length_normed) #(B,)
    
    # exponentiated cross entropy
    ece = jnp.exp(loss)
    
    out_dict = {'neg_logP_length_normed': aux_dict['neg_logP_length_normed'],
                'sum_neg_logP': aux_dict['sum_neg_logP'],
                'perplexity_perSamp': perplexity_perSamp,
                'ece': ece,
                'batch_loss': batch_loss}
    
    return out_dict


def final_eval_wrapper(dataloader, 
                       dataset, 
                       best_trainstates, 
                       eval_fn_jitted,
                       jitted_determine_alignlen_bin,
                       interms_for_tboard: dict, 
                       logfile_dir: str,
                       out_arrs_dir: str, 
                       outfile_prefix: str, 
                       tboard_writer = None,
                       **kwargs):
    return_forward_pass_outputs = interms_for_tboard['forward_pass_outputs']
    
    final_ave_loss = 0
    final_ave_loss_seqlen_normed = 0
    final_perplexity = 0
    
    for batch_idx, batch in tqdm( enumerate(dataloader), total=len(dataloader) ): 
        batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch)
        batch_max_alignlen = batch_max_alignlen.item()
        
        eval_metrics = eval_fn_jitted(batch=batch, 
                                      max_align_len=batch_max_alignlen
                                      pairhmm_trainstate=best_trainstates)
        
        
        #########################################
        ### start df; record metrics per sample #
        #########################################
        final_loglikes = dataset.retrieve_sample_names(batch[-1])
        final_loglikes['logP'] = eval_metrics['sum_neg_logP']
        final_loglikes['logP/normlength'] = eval_metrics['neg_logP_length_normed']
        final_loglikes['perplexity'] = eval_metrics['perplexity_perSamp']
        final_loglikes['dataloader_idx'] = batch[-1]
        
        num_samples_in_batch = eval_metrics['sum_neg_logP'].shape[0]
        
        # record mean values to buckets
        wf = ( num_samples_in_batch / len(dataset) )
        final_ave_loss += final_loglikes['logP'].mean() * wf
        final_ave_loss_seqlen_normed += final_loglikes['logP/normlength'].mean() * wf
        final_perplexity  += final_loglikes['perplexity'].mean() * wf

        # write dataframe
        final_loglikes.to_csv((f'{logfile_dir}/{outfile_prefix}_pt{batch_idx}_'+
                              'FINAL-LOGLIKES.tsv'), sep='\t')
    
    
    ######################
    ### POST EVAL LOOP   #
    ######################
    # extract whole-dataset performance
    final_ece = jnp.exp( final_ave_loss_seqlen_normed )
    summary_stats = {'final_ave_loss':final_ave_loss, 
                     'final_ave_loss_seqlen_normed':final_ave_loss_seqlen_normed,
                     'final_perplexity':final_perplexity,
                     'final_ece':final_ece}
    
    # write summary stats collected from final_stats_for_tboard
    #   top_layer_name has already been provided
    if tboard_writer:
        write_stats_to_tabular(flat_dict = final_stats_for_tboard,
                               writer_obj = tboard_writer)
        
    return summary_stats
    
