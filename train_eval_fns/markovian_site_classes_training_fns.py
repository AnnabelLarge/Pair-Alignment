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
import os

# flax, jax, and optax
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
                            
def train_one_batch(batch, 
                    training_rngkey,
                    pairhmm_trainstate,  
                    t_array,
                    max_align_len,
                    interms_for_tboard,
                    update_grads: bool = True,
                    **kwargs):
    
    finalpred_sow_outputs = interms_for_tboard['finalpred_sow_outputs']
    batch_aligned_mats = batch[1]
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    del batch
    
    def apply_model(pairhmm_params):
        # in training, only evaluate joint loglike i.e. use default __call__
        (loss_NLL, aux_dict), sow_dict = pairhmm_trainstate.apply_fn(variables = pairhmm_params,
                                          aligned_inputs = clipped_aligned_mats,
                                          t_array = t_array,
                                          sow_intermediates = finalpred_sow_outputs,
                                          mutable=['histograms','scalars'] if finalpred_sow_outputs else [])
        
        sow_dict = {'histograms': sow_dict.get( 'histograms', dict() ),
                    'scalars': sow_dict.get( 'scalars', dict() )
                    }
        
        aux_dict['pred_layer_metrics'] = sow_dict
        return loss_NLL, aux_dict
    
    grad_fn = jax.value_and_grad(apply_model, has_aux=True)
    (batch_loss_NLL, aux_dict), grad = grad_fn(pairhmm_trainstate.params)
    
    ### only turn this off during debug
    if update_grads:
        updates, new_opt_state = pairhmm_trainstate.tx.update(grad,
                                                           pairhmm_trainstate.opt_state,
                                                           pairhmm_trainstate.params)
        new_params = optax.apply_updates(pairhmm_trainstate.params,
                                         updates)
        new_trainstate = pairhmm_trainstate.replace(params = new_params,
                                                    opt_state = new_opt_state)
    else:
        new_trainstate = pairhmm_trainstate
    
    
    ### other metrics
    # perplexity per sample
    joint_neg_logP_length_normed = aux_dict['joint_neg_logP_length_normed']
    joint_perplexity_perSamp = jnp.exp(joint_neg_logP_length_normed)
    joint_ece = jnp.exp(batch_loss_NLL)
    
    out_dict = {'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                'joint_neg_logP': aux_dict['joint_neg_logP'],
                'joint_ece': joint_ece,
                'batch_loss': batch_loss_NLL,
                'batch_ave_joint_perpl': jnp.mean(joint_perplexity_perSamp),
                'pred_layer_metrics': aux_dict['pred_layer_metrics'],
                'finalpred_gradient': grad,
                'used_tkf_beta_approx': aux_dict['used_tkf_beta_approx']}
    
    return out_dict, new_trainstate


def eval_one_batch( batch, 
                    t_array,
                    pairhmm_trainstate,  
                    pairhmm_instance,
                    max_align_len,
                    interms_for_tboard,
                    return_all_loglikes: bool,
                    **kwargs):
    """
    WARNING: might have to pull trainstate and instance out of parted+jit 
      function
    """
    finalpred_sow_outputs = interms_for_tboard['finalpred_sow_outputs']
    batch_aligned_mats = batch[1]
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    del batch
    
    if not return_all_loglikes:
        (loss_NLL, aux_dict), sow_dict = pairhmm_trainstate.apply_fn(variables = pairhmm_trainstate.params,
                                          aligned_inputs = clipped_aligned_mats,
                                          t_array = t_array,
                                          sow_intermediates = finalpred_sow_outputs,
                                          mutable=['histograms','scalars'] if finalpred_sow_outputs else [])
    
    elif return_all_loglikes:
        aux_dict, sow_dict = pairhmm_trainstate.apply_fn(variables = pairhmm_trainstate.params,
                                          aligned_inputs = clipped_aligned_mats,
                                          t_array = t_array,
                                          sow_intermediates = finalpred_sow_outputs,
                                          mutable=['histograms','scalars'] if finalpred_sow_outputs else [],
                                          method=pairhmm_instance.calculate_all_loglikes)
        
        # specifically use joint prob for loss
        loss_NLL = jnp.mean( aux_dict['joint_neg_logP_length_normed'] )
        
    sow_dict = {'histograms': sow_dict.get( 'histograms', dict() ),
                'scalars': sow_dict.get( 'scalars', dict() )
                }
    
    ### joint probability metrics
    joint_neg_logP_length_normed = aux_dict['joint_neg_logP_length_normed']
    joint_perplexity_perSamp = jnp.exp(joint_neg_logP_length_normed)
    
    out_dict = {'joint_neg_logP': aux_dict['joint_neg_logP'],
                'joint_neg_logP_length_normed': joint_neg_logP_length_normed,
                'joint_perplexity_perSamp': joint_perplexity_perSamp,
                'pred_layer_metrics': sow_dict,
                'used_tkf_beta_approx': aux_dict['used_tkf_beta_approx']}
    
    ### other metrics
    if return_all_loglikes:
        # cond
        cond_neg_logP_length_normed = aux_dict['cond_neg_logP_length_normed']
        cond_perplexity_perSamp = jnp.exp(cond_neg_logP_length_normed)
        
        out_dict['cond_neg_logP'] = aux_dict['cond_neg_logP']
        out_dict['cond_neg_logP_length_normed'] = cond_neg_logP_length_normed
        out_dict['cond_perplexity_perSamp'] = cond_perplexity_perSamp
        
        # anc marginal
        anc_neg_logP_length_normed = aux_dict['anc_neg_logP_length_normed']
        anc_perplexity_perSamp = jnp.exp(anc_neg_logP_length_normed)
        
        out_dict['anc_neg_logP'] = aux_dict['anc_neg_logP']
        out_dict['anc_neg_logP_length_normed'] = anc_neg_logP_length_normed
        out_dict['anc_perplexity_perSamp'] = anc_perplexity_perSamp
        
        # desc marginal
        desc_neg_logP_length_normed = aux_dict['desc_neg_logP_length_normed']
        desc_perplexity_perSamp = jnp.exp(desc_neg_logP_length_normed)
        
        out_dict['desc_neg_logP'] = aux_dict['desc_neg_logP']
        out_dict['desc_neg_logP_length_normed'] = desc_neg_logP_length_normed
        out_dict['desc_perplexity_perSamp'] = desc_perplexity_perSamp
    
    return out_dict
    

def final_eval_wrapper(dataloader, 
                       dataset, 
                       eval_fn_jitted,
                       jitted_determine_alignlen_bin,
                       save_per_sample_losses: bool,
                       logfile_dir: str,
                       out_arrs_dir: str, 
                       outfile_prefix: str, 
                       tboard_writer = None,
                       **kwargs):
    """
    WARNING: might have to pull trainstate and instance out of parted+jit 
      function; if so, then make them arguments to this final function
    
    eval_fn_jitted should have already been parted by providing:
        - t_array = given time array
        - pairhmm_trainstate = best trainstate
        - pairhmm_instance = model instance
        - interms_for_tboard = (value from config)
        - return_all_loglike = True
    """
    
    summary_stats = {'joint_ave_loss': 0,
                   'joint_ave_loss_seqlen_normed': 0,
                   'joint_perplexity': 0,
                   
                   'cond_ave_loss': 0,
                   'cond_ave_loss_seqlen_normed': 0,
                   'cond_perplexity': 0,
                   
                   'anc_ave_loss': 0,
                   'anc_ave_loss_seqlen_normed': 0,
                   'anc_perplexity': 0,
                   
                   'desc_ave_loss': 0,
                   'desc_ave_loss_seqlen_normed': 0,
                   'desc_perplexity': 0,
                   }
    
    for batch_idx, batch in tqdm( enumerate(dataloader), total=len(dataloader) ): 
        batch_max_alignlen = jitted_determine_alignlen_bin(batch = batch)
        batch_max_alignlen = batch_max_alignlen.item()
        
        eval_metrics = eval_fn_jitted(batch=batch, 
                                      max_align_len=batch_max_alignlen)
        
        if eval_metrics["used_tkf_beta_approx"]:
            with open(f'{args.out_arrs_dir}/FINAL-EVAL_tkf_approx.tsv','a') as g:
                g.write(f'batch {batch_idx} of {outfile_prefix}: {eval_metrics["used_tkf_beta_approx"]}\n')
        
        
        #########################################
        ### start df; record metrics per sample #
        #########################################
        final_loglikes = dataset.retrieve_sample_names(batch[-1])
        
        for prefix in ['joint','cond','anc','desc']:
            final_loglikes[f'{prefix}_logP'] = eval_metrics[f'{prefix}_neg_logP']
            final_loglikes[f'{prefix}_logP/normlength'] = eval_metrics[f'{prefix}_neg_logP_length_normed']
            final_loglikes[f'{prefix}_perplexity'] = eval_metrics[f'{prefix}_perplexity_perSamp']
        
        final_loglikes['dataloader_idx'] = batch[-1]
        num_samples_in_batch = eval_metrics['joint_neg_logP'].shape[0]
        
        # record mean values to buckets
        wf = ( num_samples_in_batch / len(dataset) )
        
        for prefix in ['joint','cond','anc','desc']:
            to_add = final_loglikes[f'{prefix}_logP'].mean() * wf
            summary_stats[f'{prefix}_ave_loss'] += to_add
            del to_add
            
            to_add = final_loglikes[f'{prefix}_logP/normlength'].mean() * wf
            summary_stats[f'{prefix}_ave_loss_seqlen_normed'] += to_add
            del to_add
            
            to_add = final_loglikes[f'{prefix}_perplexity'].mean() * wf
            summary_stats[f'{prefix}_perplexity'] += to_add
            del to_add
            
        # write dataframe
        if save_per_sample_losses:
            final_loglikes.to_csv((f'{logfile_dir}/{outfile_prefix}_pt{batch_idx}_'+
                                  'FINAL-LOGLIKES.tsv'), sep='\t')
    
    
    ######################
    ### POST EVAL LOOP   #
    ######################
    # add ECE for all
    for prefix in ['joint','cond','anc','desc']:
        to_add = jnp.exp( summary_stats[f'{prefix}_ave_loss_seqlen_normed'] )
        summary_stats[f'{prefix}_ece'] = to_add
        del to_add
    
    return summary_stats





def label_class_posteriors(batch, 
                           t_array,
                           pairhmm_trainstate,  
                           pairhmm_instance,
                           max_align_len,
                           **kwargs):
    batch_aligned_mats = batch[1]
    clipped_aligned_mats = batch_aligned_mats[:, :max_align_len, :]
    del batch
    
    out = pairhmm_trainstate.apply_fn( variables = pairhmm_trainstate.params,
                                       aligned_inputs = clipped_aligned_mats,
                                       t_array = t_array,
                                       method=pairhmm_instance.get_class_posterior_marginals )

    return out
