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
from tqdm import tqdm

# flax, jax, and optax
import jax
import jax.numpy as jnp
from jax import config
from flax import linen as nn
import optax

from utils.tensorboard_recording_utils import (calc_stats_during_final_eval,
                                               update_stats_dict,
       
                                               
def train_one_batch(batch, 
                    training_rngkey,
                    all_trainstates,  
                    t_array,
                    max_align_len,
                    interms_for_tboard,
                    update_grads: bool = True,
                    **kwargs):
    
    finalpred_sow_outputs = interms_for_tboard['finalpred_sow_outputs']
    
    _, batch_aligned_mats, _, _ = batch
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    del batch
    
    def apply_model(pairhmm_params):
        loss_dict, sow_dict = all_trainstates.apply_fn(variables = pairhmm_params,
                                          aligned_inputs = clipped_aligned_mats,
                                          t_array = t_array,
                                          sow_intermediates = finalpred_sow_outputs,
                                          mutable=['histograms','scalars'] if finalpred_sow_outputs else [])
        
        sow_dict = {'histograms': sow_dict.get( 'histograms', dict() ),
                    'scalars': sow_dict.get( 'scalars', dict() )
                    }
        
        aux_dict = {k:v for k,v in loss_dict.items() if k != 'loss'}
        aux_dict['pred_layer_metrics'] = sow_dict
        return loss_dict['loss'], aux_dict
    
    grad_fn = jax.value_and_grad(apply_model, has_aux=True)
    (batch_loss, aux_dict), grad = grad_fn(all_trainstates.params)
    
    ### only turn this off during debug
    if update_grads:
        updates, new_opt_state = all_trainstates.tx.update(grad,
                                                           all_trainstates.opt_state,
                                                           all_trainstates.params)
        new_params = optax.apply_updates(all_trainstates.params,
                                         updates)
        new_trainstate = all_trainstates.replace(params = new_params,
                                                    opt_state = new_opt_state)
    else:
        new_trainstate = all_trainstates
    
    
    ### other metrics
    # perplexity per sample
    neg_logP_length_normed = aux_dict['neg_logP_length_normed']
    perplexity_perSamp = jnp.exp(-neg_logP_length_normed) #(B,)
    
    # exponentiated cross entropy
    ece = jnp.exp(-batch_loss)
    
    out_dict = {'neg_logP_length_normed': aux_dict['neg_logP_length_normed'],
                'sum_neg_logP': aux_dict['sum_neg_logP'],
                'ece': ece,
                'batch_loss': batch_loss,
                'batch_ave_perpl': jnp.mean(perplexity_perSamp),
                'pred_layer_metrics': aux_dict['pred_layer_metrics'] }
    
    return out_dict, new_trainstate


def eval_one_batch(batch, 
                    all_trainstates,  
                    t_array,
                    max_align_len,
                    interms_for_tboard,
                    **kwargs):
    finalpred_sow_outputs = interms_for_tboard['finalpred_sow_outputs']
    
    _, batch_aligned_mats, _, _ = batch
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    del batch
    
    loss_dict, sow_dict  = all_trainstates.apply_fn(variables = all_trainstates.params,
                                      aligned_inputs = clipped_aligned_mats,
                                      t_array = t_array,
                                      sow_intermediates = finalpred_sow_outputs,
                                      mutable=['histograms','scalars'] if finalpred_sow_outputs else [])
    
    sow_dict = {'histograms': sow_dict.get( 'histograms', dict() ),
                'scalars': sow_dict.get( 'scalars', dict() )
                }
    
    ### other metrics
    # perplexity per sample
    neg_logP_length_normed = loss_dict['neg_logP_length_normed']
    perplexity_perSamp = jnp.exp(-neg_logP_length_normed) #(B,)
    
    # exponentiated cross entropy
    ece = jnp.exp(-loss_dict['loss'])
    
    out_dict = {'neg_logP_length_normed': loss_dict['neg_logP_length_normed'],
                'sum_neg_logP': loss_dict['sum_neg_logP'],
                'perplexity_perSamp': perplexity_perSamp,
                'ece': ece,
                'batch_loss': loss_dict['loss'],
                'pred_layer_metrics': sow_dict}
    out_dict = {**out_dict, **sow_dict}
    return out_dict


def final_eval_wrapper(dataloader, 
                       dataset, 
                       best_trainstates, 
                       eval_fn_jitted,
                       jitted_determine_alignlen_bin,
                       save_per_sample_losses: bool,
                       logfile_dir: str,
                       out_arrs_dir: str, 
                       outfile_prefix: str, 
                       tboard_writer = None,
                       **kwargs):
    final_ave_loss = 0
    final_ave_loss_seqlen_normed = 0
    final_perplexity = 0
    
    for batch_idx, batch in tqdm( enumerate(dataloader), total=len(dataloader) ): 
        batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch)
        batch_max_alignlen = batch_max_alignlen.item()
        
        eval_metrics = eval_fn_jitted(batch=batch, 
                                      max_align_len=batch_max_alignlen,
                                      all_trainstates=best_trainstates)
        
        
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
        if save_per_sample_losses:
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
      
    return summary_stats
    
