#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 16:57:27 2025

@author: annabel
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from models.latent_class_mixtures.one_dim_backward_joint_loglikes import joint_only_one_dim_backward as joint_only_backward
from models.latent_class_mixtures.one_dim_forward_joint_loglikes import joint_only_one_dim_forward as joint_only_forward

def forward_backward_posteriors(aligned_inputs,
                                joint_logprob_emit_at_match,
                                logprob_emit_at_indel,
                                joint_logprob_transit,
                                unique_time_per_sample: bool, 
                                return_checksum: bool=False):
    B = aligned_inputs.shape[0]
    C = logprob_emit_at_indel.shape[0]
    mask = (aligned_inputs[...,0]!=0) #(B,L)
    mask = mask.T #(L, B)
    
    
    ### forward and backward
    fwd = joint_only_forward(aligned_inputs = aligned_inputs,
                             joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                             logprob_emit_at_indel = logprob_emit_at_indel,
                             joint_logprob_transit = joint_logprob_transit,
                             unique_time_per_sample = unique_time_per_sample,
                             return_all_intermeds = True) #(L-1, T, C, B) or #(L-1, C, B)
    
    bkw = joint_only_backward(aligned_inputs = aligned_inputs,
                              joint_logprob_emit_at_match = joint_logprob_emit_at_match,
                              logprob_emit_at_indel = logprob_emit_at_indel,
                              joint_logprob_transit = joint_logprob_transit,
                              unique_time_per_sample = unique_time_per_sample) #(L-1, T, C, B) or #(L-1, C, B)
    
    
    ### combine, mask
    # unique branch length per sample
    if unique_time_per_sample:
        mask = mask[:,None,None,:] #(L, 1, B)
        fwd = jnp.concatenate( [jnp.zeros( (1, C, B) ), fwd], axis=0 ) #(L, C, B) 
        bkw = jnp.concatenate( [bkw, jnp.zeros( (1, C, B) )], axis=0 ) #(L, C, B) 
    
    # marginalize over time grid
    elif not unique_time_per_sample:
        T = joint_logprob_transit.shape[0]
        mask = mask[:,None,None,:] #(L, 1, 1, B)
        fwd = jnp.concatenate( [jnp.zeros( (1, T, C, B) ), fwd], axis=0 ) #(L, T, C, B) 
        bkw = jnp.concatenate( [bkw, jnp.zeros( (1, T, C, B) )], axis=0 ) #(L, T, C, B) 
        
    # add, mask
    total_post = fwd + bkw #(L, T, C, B) or #(L-1, C, B)
    total_post = jnp.multiply( total_post, mask ) #(L, T, C, B) or #(L-1, C, B)
    
    if return_checksum:
        fwd_sum = logsumexp(fwd[-1,...], axis=1) #(T, B) or (B,)
        bkw_sum = logsumexp(bkw[0,...], axis=1) #(T, B) or (B,)
        check = jnp.abs( fwd_sum - bkw_sum ) #(T, B) or (B,)
        
        return total_post, check
    
    else:
        return total_post
    
    
